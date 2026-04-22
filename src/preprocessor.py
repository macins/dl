import polars as pl

class CustomScaler:
    def __init__(self, features=[], fillna=True, clip=True, demean=True, scale=False, verbose=False):
        self.features = features
        self.fillna = fillna
        self.clip = clip and fillna
        self.demean = demean and clip
        self.scale = scale and demean
        self.verbose = verbose
        self.stats = {}
        if self.verbose:
            print("Initailzed successfully!")
        
    def set_features(self, features):
        self.features = features

    def fit_transform(self, df):
        if self.verbose:
            print("Start fitting...")
        if self.fillna:
            if self.verbose:
                print("Filling nan...")
            df = df.fill_nan(None).fill_null(0)

        if self.clip:
            if self.verbose:
                print("Calculating stats...")
            long = df.select([
                pl.struct([
                    pl.col(c).mean().alias("mean"),
                    pl.col(c).std().alias("std"),
                    pl.col(c).min().alias("min"),
                    pl.col(c).max().alias("max"),
                    pl.min_horizontal(pl.col(c).mean() + 4.9 * pl.col(c).std()).alias("ceil"),
                    pl.max_horizontal(pl.col(c).mean() - 4.9 * pl.col(c).std()).alias("floor")
                ]).alias(c)
                for c in self.features
            ])
            to_clip = self.features
            for k, v in long.to_dict().items():
                self.stats[k] = v.item()
            i = 0
            
            if self.verbose:
                print(self.stats)
                
            while to_clip:
                if i > 500:
                    break
                i += 1
                if self.verbose:
                    print(f"The {i}-th round...")
                df = df.with_columns([
                    pl.col(c).clip(self.stats[c]["floor"], self.stats[c]["ceil"]).alias(c)
                    for c in to_clip
                ])
                long = df.select([
                    pl.struct([
                        pl.col(c).mean().alias("mean"),
                        pl.col(c).std().alias("std"),
                        pl.col(c).min().alias("min"),
                        pl.col(c).max().alias("max"),
                        (pl.col(c).mean() + 4.9 * pl.col(c).std()).alias("ceil"),
                        (pl.col(c).mean() - 4.9 * pl.col(c).std()).alias("floor")
                    ]).alias(c)
                    for c in to_clip
                ])
                for k, v in long.to_dict().items():
                    self.stats[k] = v.item()
                to_clip = [
                    c for c in to_clip if
                    (self.stats[c]["min"] < self.stats[c]["mean"] - 5 * self.stats[c]["std"]) or 
                    (self.stats[c]["max"] > self.stats[c]["mean"] + 5 * self.stats[c]["std"])
                ]

        if self.demean:
            df = df.with_columns([
                (pl.col(c) - pl.col(c).mean()).alias(c)
                for c in self.features
            ])
        
        if self.scale:
            df = df.with_columns([
                (
                    pl.col(c) / (pl.col(c).std() + 1e-12)
                ).alias(c)
                for c in self.features
            ])

        return df.fill_null(0)

    def transform(self, df):
        if self.fillna:
            df = df.fill_nan(None).fill_null(0)
        
        if self.clip:
            df = df.with_columns([
                pl.col(c).clip(self.stats[c]["floor"], self.stats[c]["ceil"]).alias(c)
                    for c in self.features
                ])

        if self.demean:
            df = df.with_columns([
                (pl.col(c) - self.stats[c]["mean"]).alias(c)
                for c in self.features
            ])
        
        if self.scale:
            df = df.with_columns([
                (
                    pl.col(c) / (self.stats[c]["std"] + 1e-12)
                ).alias(c)
                for c in self.features
            ])

        return df.fill_null(0)
                    