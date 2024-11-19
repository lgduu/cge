# +
import pdb

from typing import Union, List
from transformers import LlamaTokenizer, BatchEncoding


# +
class OpenLlamaTokenizer(LlamaTokenizer):
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            if token == '▁': continue
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids
    
#     def batch_encode_plus(
#         self,
#         *args,
#         **kwargs,
#     ) -> BatchEncoding:
#         encodings = super().batch_encode_plus(*args, **kwargs)
#         indices = encodings.input_ids != 31822 # remove blank token
#         if torch.any(~indices):
#             encodings = BatchEncoding({
#                 key: tensor[indices]
#                 for key, tensor in encodings.items()
#             })
#         return encodings
