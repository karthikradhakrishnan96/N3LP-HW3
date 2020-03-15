import torch
from sesame_street.models import Bert
import os
import json

class BertWeightLoader():
    def __init__(self):
        pass

    @staticmethod
    def load_embed_weights(model, weights, mapping):
        new_json = {}
        for key in model.state_dict():
            if key not in mapping:
                print("Skipping ", key)
            else:
                map_value = mapping[key]
                new_json[key] = weights[map_value]

        model.load_state_dict(new_json)

    @staticmethod
    def load_encoder_stack_weights(model, weights, mapping):
        for i, encoder in enumerate(model.encoder_stack.layers):
            root_name = mapping["root_name"] + str(i) + "."
            weight = {k: weights[k] for k in weights if root_name in k}
            BertWeightLoader.load_encoder_weights(encoder, weight, mapping, root_name)

    @staticmethod
    def load_encoder_weights(model, weights, mapping, root_name):
        new_json = {}
        attention_map = mapping["attention"]
        other_map = mapping["other"]

        for key in model.self_attn.state_dict():
            if key not in attention_map:
                print("Skipping ", key)
            else:
                map_value = attention_map[key]
                if isinstance(map_value, list):
                    values = [weights[root_name + x] for x in map_value]
                    map_weights = torch.cat(values, dim=0)
                elif isinstance(map_value, str):
                    map_weights = weights[root_name + map_value]
                else:
                    print("Not list or str for key ", key)

                new_json["self_attn." + key] = map_weights

        for key in model.state_dict():
            if key in new_json:
                continue
            if key not in other_map:
                print("Skipping ", key)
            else:
                map_value = other_map[key]
                new_json[key] = weights[root_name + map_value]

        model.load_state_dict(new_json)

    @staticmethod
    def load_condenser_weights(model, weights, mapping):
        new_json = {}
        for key in model.state_dict():
            if key not in mapping:
                print("Skipping ", key)
            else:
                new_json[key] = weights[mapping[key]]

        model.load_state_dict(new_json)

    @staticmethod
    def from_hugging_face(model, weights):
        embed_weights = {k: weights[k] for k in weights if "embeddings" in k}
        encoder_weights = {k: weights[k] for k in weights if "encoder" in k}
        condenser_weights = {k: weights[k] for k in weights if "pooler" in k}
        # TODO : Verify where do cls.predictions.X weights come from
        # TODO : Verify where does cls.seq_relationship weight come from

        with open(os.path.sep.join(["constants", "hugging_face_weight_map.json"])) as f:
            weight_map = json.load(f)

        children = [child for child in model.children()]
        BertWeightLoader.load_embed_weights(children[0], embed_weights, weight_map["embed"])
        BertWeightLoader.load_encoder_stack_weights(children[1], encoder_weights, weight_map["encoder"])
        BertWeightLoader.load_condenser_weights(children[2], condenser_weights, weight_map["condenser"])