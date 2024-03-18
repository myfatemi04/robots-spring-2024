from transformers.models.clip.modeling_clip import CLIPVisionTransformer, BaseModelOutputWithPooling
from typing import Optional, Union, Tuple

# Modify the CLIP forward pass to accept our special "plan location" token as an input.
def custom_clip_vision_transformer_forward(
    self: CLIPVisionTransformer,
    pixel_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    plan_location: Optional[torch.FloatTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    hidden_states = self.embeddings(pixel_values)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
