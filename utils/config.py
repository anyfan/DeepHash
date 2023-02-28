def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    config["data"] = {
        "train_set": {
            "list_path": "./data/" + config["dataset"] + "/train.txt",
            "batch_size": config["batch_size"]
        },
        "database": {
            "list_path": "./data/" + config["dataset"] + "/database.txt",
            "batch_size": config["batch_size"]
        },
        "test": {
            "list_path": "./data/" + config["dataset"] + "/test.txt",
            "batch_size": config["batch_size"]
        }
    }
    return config