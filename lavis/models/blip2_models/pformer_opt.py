import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer, BertTokenizer
from lavis.models.blip2_models.modeling_bert import BertConfig, BertModel

import copy   ### for OPT vocabulary embeddings

@registry.register_model("pformer_opt")
class PformerOPT(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pformer_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",  ### TODO
        "pformer_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",  ### TODO
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
    ):
        super().__init__()

        # Added hyper-parameters
        self.embed_dim = 256   # embed dimension for self-contrastive learning

        # Promptformer (Pformer)
        self.Pformer = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.Pformer.cls_proj = nn.Linear(self.Pformer.config.hidden_size, num_query_token*self.Pformer.config.hidden_size)
        self.Pformer.pooler = nn.Sequential(nn.Linear(self.Pformer.config.hidden_size, self.Pformer.config.hidden_size), nn.Tanh())
        self.Pformer.text_proj = nn.Linear(self.Pformer.config.hidden_size, self.embed_dim) # projection layer for self-contrastive learning

        self.Pformer.temp = nn.Parameter(0.07 * torch.ones([]))

        # OPT model
        self.opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16)
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False   ### freezing all parameters in OPT

        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.opt_tokenizer.padding_side = "right"
        self.register_buffer("opt_vocab_embeddings", copy.copy(self.opt_model.model.decoder.embed_tokens.weight.data))

        self.Pformer.opt_proj = nn.Linear(
            self.Pformer.config.hidden_size, self.opt_model.config.hidden_size
        )   ### project BERT hidden size to OPT hidden size, e.g., 768 -> 2560

        # retained from BLIP2OPT
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        text = [t + "\n" for t in samples["text_input"]]

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.Pformer.device)

        ###### Mask 25% of inputs ######
        input_ids = text_tokens.input_ids.clone()
        probability_matrix = torch.full(input_ids.shape, 0.25)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        input_ids[masked_indices] = self.tokenizer.mask_token_id
        text_tokens.input_ids = input_ids
        ################################

        text_output = self.Pformer.forward(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        ###============== Self-contrastive loss ===================###
        text_feat = F.normalize(
            self.Pformer.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_t2t = torch.matmul(
            text_feat, text_feat_all.t()
        )
        sim_t2t = sim_t2t / self.Pformer.temp

        rank = dist.get_rank()
        bs = text_feat.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            self.Pformer.device
        )

        loss_c = (
            F.cross_entropy(sim_t2t, targets, label_smoothing=0.1)
        )

        ###============== Quantization loss ===================###
        bsz, n_query, hdim = text_output.last_hidden_state.shape
        inputs_opt = self.Pformer.cls_proj(text_output.last_hidden_state[:, 0, :])   ### 1x768 --> 1x(32x768)
        inputs_opt = inputs_opt.reshape(bsz, n_query, hdim)   ### 1x(32x768) --> 1x32x768
        inputs_opt = self.Pformer.pooler(inputs_opt)
        inputs_opt = self.Pformer.opt_proj(inputs_opt)  ### 768 --> 2560
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(inputs_opt.device)

        inputs_opt_flatten = inputs_opt.reshape(-1, inputs_opt.shape[-1])
        distances = (torch.sum(inputs_opt_flatten**2, dim=1, keepdim=True)
                    + torch.sum(self.opt_vocab_embeddings**2, dim=1)
                    - 2 * torch.matmul(inputs_opt_flatten, self.opt_vocab_embeddings.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.opt_vocab_embeddings.shape[0]).to(inputs_opt.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.opt_vocab_embeddings)
        loss_vq = F.mse_loss(quantized.detach(), inputs_opt_flatten)

        ###============== Language modeling loss ===================###
        self.opt_tokenizer.padding_side = "right"
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.opt_model.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(self.opt_model.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss_lm = outputs.loss

        loss = loss_c + loss_vq + loss_lm
        return {"loss": loss, "loss_c": loss_c, "loss_vq": loss_vq, "loss_lm": loss_lm}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
            text = [t + "\n" for t in samples["text_input"]]

            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.Pformer.device)

            text_output = self.Pformer.forward(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )

            bsz, n_query, hdim = text_output.last_hidden_state.shape
            inputs_opt = self.Pformer.cls_proj(text_output.last_hidden_state[:, 0, :])   ### 1x768 --> 1x(32x768)
            inputs_opt = inputs_opt.reshape(bsz, n_query, hdim)   ### 1x(32x768) --> 1x32x768
            inputs_opt = self.Pformer.pooler(inputs_opt)
            inputs_opt = self.Pformer.opt_proj(inputs_opt)  ### 768 --> 2560
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(inputs_opt.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * inputs_opt.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(self.opt_model.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):

        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        # model.load_checkpoint_from_config(cfg) ##### No pretrained weights

        return model
