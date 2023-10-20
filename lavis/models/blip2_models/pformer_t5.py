import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers import AutoTokenizer, BertTokenizer
from lavis.models.blip2_models.modeling_bert import BertConfig, BertModel

import copy   ### for T5 vocabulary embeddings
import random

@registry.register_model("pformer_t5")
class PformerT5(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pformer_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",       ### TODO
        "pformer_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",     ### TODO
    }

    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        # Added hyper-parameters
        self.embed_dim = 256   # embed dimension for self-contrastive learning

        # Pformer
        self.Pformer = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.Pformer.cls_proj = nn.Linear(self.Pformer.config.hidden_size, num_query_token*self.Pformer.config.hidden_size)
        self.Pformer.pooler = nn.Sequential(nn.Linear(self.Pformer.config.hidden_size, self.Pformer.config.hidden_size), nn.Tanh())
        self.Pformer.text_proj = nn.Linear(self.Pformer.config.hidden_size, self.embed_dim) # projection layer for self-contrastive learning

        self.Pformer.temp = nn.Parameter(0.07 * torch.ones([]))

        # T5 model
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()   ### freezing all parameters in T5

        self.register_buffer("t5_vocab_embeddings", copy.copy(self.t5_model.shared.weight.data.float()))

        self.Pformer.t5_proj = nn.Linear(
            self.Pformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):

        ### new code starts
        text = samples["text_input"]
        text_input, text_output = [], []
        for t in text:
            words = t.split(" ")
            mid_point = random.randint(1, 1+len(words)//2)
            text_input.append(" ".join(words[:mid_point]))
            text_output.append(" ".join(words[mid_point:]))
        samples["text_input"] = text_input
        samples["text_output"] = text_output
        ### new code ends

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
        inputs_t5 = self.Pformer.cls_proj(text_output.last_hidden_state[:, 0, :])   ### 1x768 --> 1x(32x768)
        inputs_t5 = inputs_t5.reshape(bsz, n_query, hdim)   ### 1x(32x768) --> 1x32x768
        inputs_t5 = self.Pformer.pooler(inputs_t5)
        inputs_t5 = self.Pformer.t5_proj(inputs_t5)  ### 768 --> 2560
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(inputs_t5.device)

        # inputs_t5_flatten = inputs_t5.reshape(-1, inputs_t5.shape[-1])
        # distances = (torch.sum(inputs_t5_flatten**2, dim=1, keepdim=True)
        #             + torch.sum(self.t5_vocab_embeddings**2, dim=1)
        #             - 2 * torch.matmul(inputs_t5_flatten, self.t5_vocab_embeddings.t()))
        #
        # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encodings = torch.zeros(encoding_indices.shape[0], self.t5_vocab_embeddings.shape[0]).to(inputs_t5.device)
        # encodings.scatter_(1, encoding_indices, 1)
        #
        # # Quantize and unflatten
        # quantized = torch.matmul(encodings, self.t5_vocab_embeddings)
        # loss_vq = F.mse_loss(quantized.detach(), inputs_t5_flatten)

        ###============== Language modeling loss ===================###
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.t5_model.device)
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.t5_model.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss_lm = outputs.loss

        loss = loss_c + loss_lm
        return {"loss": loss, "loss_c": loss_c, "loss_lm": loss_lm}

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
            text = [ " ".join(t) for t in zip(samples["text_input"], samples["text_output"]) ]

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
            inputs_t5 = self.Pformer.cls_proj(text_output.last_hidden_state[:, 0, :])   ### 1x768 --> 1x(32x768)
            inputs_t5 = inputs_t5.reshape(bsz, n_query, hdim)   ### 1x(32x768) --> 1x32x768
            inputs_t5 = self.Pformer.pooler(inputs_t5)
            inputs_t5 = self.Pformer.t5_proj(inputs_t5)  ### 768 --> 2560
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(inputs_t5.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            if isinstance(prompt, str):
                prompt = [prompt] * bsz
            else:
                assert len(prompt) == bsz, "The number of prompts must be equal to the batch size."

            input_tokens = self.t5_tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).to(inputs_t5.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

                outputs = self.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
                output_text = self.t5_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
            return output_text

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        # model.load_checkpoint_from_config(cfg)

        return model
