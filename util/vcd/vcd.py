import re

import pandas as pd
from vcd.reader import ScalarChange, TokenKind, VarDecl, VectorChange, tokenize


def parse_vcd_to_df(vcd_file: str) -> pd.DataFrame:
    identifiers: dict[str, str] = dict()
    current_frame: dict[str, int | str] = dict()

    clock_id = None

    frames = []

    patterns = [
        r"clock",  # clock
        r"^io_data_tcdm_req_\d+_.+$",  # all tcdm requests
    ]

    with open(vcd_file, "rb") as f:
        for token in tokenize(f):
            if token.kind == TokenKind.VAR:
                # variable declaration
                # declares variable id_code to its reference
                assert isinstance(token.data, VarDecl)

                # value must match one of the patterns
                if not any(
                    [re.match(pattern, token.data.reference) for pattern in patterns]
                ):
                    continue
                identifiers[token.data.id_code] = token.data.reference
                current_frame[token.data.id_code] = 0

                if token.data.reference == "clock":
                    clock_id = token.data.id_code

            elif token.kind == TokenKind.CHANGE_SCALAR:
                # scalar change of variable
                assert isinstance(token.data, ScalarChange)

                # only save tracking values
                if token.data.id_code not in identifiers:
                    continue

                current_frame[token.data.id_code] = int(token.data.value)

                if token.data.id_code == clock_id and int(token.data.value) == 1:
                    # rising clock cycle
                    # output current frame
                    frames.append(current_frame.copy())

            elif token.kind == TokenKind.CHANGE_VECTOR:
                # vector change of variable
                assert isinstance(token.data, VectorChange)

                # only save tracking values
                if token.data.id_code not in identifiers:
                    continue
                current_frame[token.data.id_code] = token.data.value

        breakpoint()

    # create pandas dataframe from frames
    df = pd.DataFrame(frames, columns=frames[0].keys())
    df = df.rename(columns=identifiers)

    return df
