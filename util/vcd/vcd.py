import os
from os.path import join

import pandas as pd
from vcd.reader import TokenKind, VarDecl, tokenize


def parse_vcd_to_df(vcd_file: str) -> pd.DataFrame:

    time = 0

    identifiers = dict()
    current_frame = dict()

    clock_id = None

    frames = []

    with open(vcd_file, 'rb') as f:

        tokens = tokenize(f)
        for token in tokens:

            # unused tokens
            if token.kind == TokenKind.DATE:
                continue
            elif token.kind == TokenKind.VERSION:
                continue
            elif token.kind == TokenKind.COMMENT:
                continue
            elif token.kind == TokenKind.TIMESCALE:
                continue
            elif token.kind == TokenKind.SCOPE:
                continue
            elif token.kind == TokenKind.COMMENT:
                continue
            elif token.kind == TokenKind.UPSCOPE:
                continue
            elif token.kind == TokenKind.DUMPON:
                continue
            elif token.kind == TokenKind.DUMPOFF:
                continue
            elif token.kind == TokenKind.ENDDEFINITIONS:
                continue
            elif token.kind == TokenKind.DUMPVARS:
                continue


            elif token.kind == TokenKind.VAR:
                assert isinstance(token.data, VarDecl)
                identifiers[token.data.id_code] = token.data.reference
                current_frame[token.data.id_code] = 0

                if token.data.reference == 'clock':
                    clock_id = token.data.id_code

            elif token.kind == TokenKind.CHANGE_TIME:
                # print(f'change time fron {time} to {token.data}')
                time = token.data

            elif token.kind == TokenKind.CHANGE_SCALAR:
                current_frame[token.data.id_code] = int(token.data.value)

                if token.data.id_code == clock_id and int(token.data.value) == 1:
                    # rising clock cycle
                    # output current frame
                    frames.append(current_frame.copy())

            elif token.kind == TokenKind.CHANGE_VECTOR:
                current_frame[token.data.id_code] = token.data.value

            else:
                pass


    # create pandas dataframe from frames
    df = pd.DataFrame(frames, columns = frames[0].keys())
    df = df.rename(columns=identifiers)

    return df
