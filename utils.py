import json
import git
import os
import time

class DotDict(dict):
    '''
        dot.notation access to dictionary attributes

        Inputs: dict: dict
            - dict: a non-empty dict
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delattr__

def filter_dotdict_class_propoperty(dotdict_object, class_blueprint):
    '''
        Generate new dotdict with properties exist in class_blueprint
        
        Inputs: dotdict_object (dotdict), class_blueprint (Object)
            - dotdict_object: an instance of dotdict
            - class_blueprint: a class where its properties act as filter for dotdict_object

        Outputs: dotdict_object: dotdict

    '''
    varnames = class_blueprint.__init__.__code__.co_varnames
    new_dotdict = DotDict({k:v for (k, v) in dotdict_object.items() if k in varnames})
    return new_dotdict

def load_training_parameter_from_json(path):
    '''
        Load parameters from json file to DotDict.

        Inputs: path (str)
            - path: path of the json file
        
        Outputs: dotdict_object (DotDict)
            - dotdict_object: loaded parameter
    '''

    json_object = json.load(open(path, "r"))
    dotdict_object = DotDict(json_object)

    # use git versioning
    if dotdict_object.use_git:
        repo = git.Repo(os.getcwd())
        headcommit = repo.head.commit
        current_branch = repo.active_branch.name

        #get current branch + time stamp
        version = current_branch + "_commited_at_" + time.strftime("%a_%d_%b_%Y_%H_%M", time.gmtime(headcommit.committed_date))
        
        # new result folder path
        dotdict_object.result_folder_path = os.path.join(dotdict_object.result_folder_path, version)

    return dotdict_object

def load_rest_api_parameter_from_json(path):
    '''
        Load parameters from json file to DotDict.

        Inputs: path (str)
            - path: path of the json file
        
        Outputs: dotdict_object (DotDict)
            - dotdict_object: loaded parameter
    '''
    json_object = json.load(open(path, "r"))
    dotdict_object = DotDict(json_object)

    return dotdict_object

class FactoryClass:
    '''
        Class for generating an intance of a class

        Inputs: class_constructor (Class), param_dict (dict)
            - class_constructor: a class that will be used to create instance
            - param_dict: a dict contains parameter for the class_constructor
    '''
    def __init__(self, class_contructor, param_dict = {}):

        self.class_contructor = class_contructor
        self.param_dict = param_dict
    
    def create_class(self, new_param_dict={}):
        '''
            Create new instance of class constructor

            Inputs: new_param_dict (dict)
                - new_param_dict : dict of new parameter that is not available at param_dict
        '''
        new_object = self.class_contructor(**self.param_dict, **new_param_dict)
        return new_object
