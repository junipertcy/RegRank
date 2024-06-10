from ._registry import registry, registry_urls

# import networkx as nx
import graph_tool.all as gt
import zstandard as zstd
from pathlib import Path
import tempfile

try:
    import pooch
except ImportError:
    pooch = None
    data_fetcher = None
else:
    data_fetcher = pooch.create(
        # Use the default cache folder for the operating system
        # Pooch uses appdirs (https://github.com/ActiveState/appdirs) to
        # select an appropriate directory for the cache on each platform.
        path=pooch.os_cache("rSpringRank-data"),
        # The remote data is on Github
        # base_url is a required param, even though we override this
        # using individual urls in the registry.
        base_url="https://github.com/junipertcy/",
        registry=registry,
        urls=registry_urls,
    )


def fetch_data(dataset_name, data_fetcher=data_fetcher):
    if data_fetcher is None:
        raise ImportError(
            "Missing optional dependency 'pooch' required "
            "for scipy.datasets module. Please use pip or "
            "conda to install 'pooch'."
        )
    # The "fetch" method returns the full path to the downloaded data file.
    return data_fetcher.fetch(dataset_name)


def us_air_traffic():
    # The file will be downloaded automatically the first time this is run,
    # returning the path to the downloaded file. Afterwards, Pooch finds
    # it in the local cache and doesn't repeat the download.
    fname = fetch_data("us_air_traffic.gt.zst")
    # Now we just need to load it with our standard Python tools.
    fname = Path(fname)
    dctx = zstd.ZstdDecompressor()

    with tempfile.TemporaryDirectory(delete=True) as temp_dir:
        temp_dir_path = Path(temp_dir)
        gt_fname = temp_dir_path / fname.with_suffix("").name
        with open(fname, "rb") as ifh, open(gt_fname, "wb") as ofh:
            dctx.copy_stream(ifh, ofh)
        graph = gt.load_graph(gt_fname.as_posix())
        return graph
