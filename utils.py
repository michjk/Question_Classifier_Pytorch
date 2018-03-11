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

def load_parameters_from_json(path):
    json_object = json.load(open(path, "r"))
    dotdict_object = dotdict(json_object)

    if dotdict_object.use_git:
        repo = git.Repo(os.getcwd())
        headcommit = repo.head.commit
        current_branch = repo.active_branch.name
        dotdict_object.result_folder_path = os.path.join(dotdict_object.result_folder_path, current_branch + "_" + time.strftime("%a_%d_%b_%Y_%H_%M", time.gmtime(headcommit.committed_date)))

    return dotdict_object