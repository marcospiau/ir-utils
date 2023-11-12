import polars as pl

def pack_polars_dataframe(df, col):
    """Pack a dataframe into a single column and row."""
    return df.groupby(pl.lit(1)).agg(
            pl.struct('*').alias(col)).select(col)

def unpack_polars_dataframe(df, col):
    """Unpack a dataframe from a single column and row."""
    return df.select(pl.col(col).explode()).unnest(col)