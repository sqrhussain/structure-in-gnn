
# A slightly modified version of James Allen's code, to split a list of numbers in the format [x1,...,xn]
# Source: https://gist.github.com/jlln/338b4b0b55bd6984f883

import pandas as pd

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row_string = row[target_column][1:-1].split(separator) # [1:-1] to ignore brackets 
        split_row = [float(x) for x in split_row_string]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

