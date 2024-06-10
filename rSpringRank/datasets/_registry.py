##############################################################################
# This file serves as the dataset registry for rSpringRank Datasets SubModule.
##############################################################################


# To generate the SHA256 hash, use the command
# openssl sha256 <filename>
registry = {
    "us_air_traffic.gt.zst": "433c8d1473530a40c747c56159c9aee8a8cd404b9585704c41f57e43383d3187"
}

registry_urls = {
    "us_air_traffic.gt.zst": "https://networks.skewed.de/net/us_air_traffic/files/us_air_traffic.gt.zst"
}

# dataset method mapping with their associated filenames
# <method_name> : ["filename1", "filename2", ...]
method_files_map = {
    "us_air_traffic": ["us_air_traffic.gt.zst"]
}