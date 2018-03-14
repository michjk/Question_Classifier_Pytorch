import json
import git
import os
import time

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delattr__

def filter_dotdict_class_propoperty(dotdict_object, class_blueprint):
    varnames = class_blueprint.__init__.__code__.co_varnames
    new_dotdict = {k:v for (k, v) in dotdict_object.items() if k in varnames}
    return new_dotdict

def load_parameter_from_json(path):
    json_object = json.load(open(path, "r"))
    dotdict_object = dotdict(json_object)

    if dotdict_object.use_git:
        repo = git.Repo(os.getcwd())
        headcommit = repo.head.commit
        current_branch = repo.active_branch.name
        version = current_branch + "_commited_at_" + time.strftime("%a_%d_%b_%Y_%H_%M", time.gmtime(headcommit.committed_date)) + "_run_at_" + str(int(time.time() - headcommit.committed_date))
        dotdict_object.result_folder_path = os.path.join(dotdict_object.result_folder_path, version)

    return dotdict_object

def load_rest_api_parameter_from_json(path):
    json_object = json.load(open(path, "r"))
    dotdict_object = dotdict(json_object)

    return dotdict_object