import pyrebase

# Your credentials after create a app web project.
config = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

# Upload
path_on_cloud = "sequential-1/model.json"
path_local = "sequential-models/sequential-1/model.json"
storage.child(path_on_cloud).put(path_local)

# Upload 2
path_on_cloud = "sequential-1/group1-shard1of1.bin"
path_local = "sequential-models/sequential-1/group1-shard1of1.bin"
storage.child(path_on_cloud).put(path_local)

# Download
# storage.child(path_on_cloud).download("<file_downloaded>")
