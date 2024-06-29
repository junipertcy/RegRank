##############################################################################
# This file serves as the dataset registry for rSpringRank Datasets SubModule.
##############################################################################


# To generate the SHA256 hash, use the command
# openssl sha256 <filename>
registry = {
    "us_air_traffic.gt.zst": "433c8d1473530a40c747c56159c9aee8a8cd404b9585704c41f57e43383d3187",
    "PhD_exchange.txt": "2ec743facc4c0e3a6e17ffca4cf321b7543c7d176061325cfbd7d8e7bf79eea7",
    "school_names.txt": "aeaf5bcfc8a9b4f374c1e1d8c901fd5ce6781ba1673d5a1556dd95d845131670",
    "at_migrations.gt.zst": "01485db4d03cb5339892375daf9cce20f171f02d2ab46ae9ee9d31e21746921b",
}

registry_urls = {
    "PhD_exchange.txt": "https://raw.githubusercontent.com/junipertcy/rSpringRank-data/main/PhD_exchange/PhD_exchange.txt",
    "school_names.txt": "https://raw.githubusercontent.com/junipertcy/rSpringRank-data/main/PhD_exchange/school_names.txt",
    "us_air_traffic.gt.zst": "https://networks.skewed.de/net/us_air_traffic/files/us_air_traffic.gt.zst",
    "at_migrations.gt.zst": "https://networks.skewed.de/net/at_migrations/files/at_migrations.gt.zst",
}

# dataset method mapping with their associated filenames
# <method_name> : ["filename1", "filename2", ...]
method_files_map = {
    "PhD_exchange": ["PhD_exchange.txt"],
    "PhD_exchange_school_names": ["school_names.txt"],
    "us_air_traffic": ["us_air_traffic.gt.zst"],
    "at_migrations": ["at_migrations.gt.zst"],
}
