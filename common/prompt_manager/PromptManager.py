import os
import shutil
import warnings

class Model:
    def __init__(self, name, main_folder_path='uat_eval/common/prompt_manager/prompts'):
        self.name = name
        self.main_folder_path = main_folder_path
        self.folder_path = os.path.join(main_folder_path, name)
        os.makedirs(self.folder_path, exist_ok=True)
        self.prompts = {}
        self.update_prompts()

    def add_prompt(self, name, content):
        """Add a prompt to the model.

        Args:
            name (str): The name of the prompt.
            content (str): The content of the prompt.
        """
        prompt_folder_path = os.path.join(self.folder_path, name)
        os.makedirs(prompt_folder_path, exist_ok=True)
        with open(os.path.join(prompt_folder_path, 'system.txt'), 'w') as file:
            file.write(content)
        self.prompts[name] = content

    def get_prompt(self, name, context=None):
        """Get the content of a prompt with the given context.

        Args:
            name (str): The name of the prompt.
            context (str): The context to use when formatting the prompt.

        Returns:
            str: The content of the prompt with the given context.
        """
        prompt = self.prompts.get(name, 'Prompt not found')
        if prompt != 'Prompt not found':
            return prompt.format(context=context)
        else:
            return prompt

    def get_prompts(self):
        """Get a list of the names of the prompts in the model.

        Returns:
            list: A list of the names of the prompts in the model.
        """
        return list(self.prompts.keys())

    def delete_prompt(self, name):
        """Delete a prompt from the model.

        Args:
            name (str): The name of the prompt to delete.
        """
        prompt_folder_path = os.path.join(self.folder_path, name)
        if os.path.exists(prompt_folder_path):
            shutil.rmtree(prompt_folder_path)
            del self.prompts[name]
        else:
            warnings.warn(f"Prompt '{name}' does not exist.")

    def update_prompts(self):
        """Update the prompts in the model based on the folders in the model folder path."""
        for folder_name in os.listdir(self.folder_path):
            folder_path = os.path.join(self.folder_path, folder_name)
            if os.path.isdir(folder_path):
                with open(os.path.join(folder_path, 'system.txt'), 'r') as file:
                    self.prompts[folder_name] = file.read()

    def __str__(self):
        """Return a string representation of the model.

        Returns:
            str: A string representation of the model, including the name of the model and the names of the prompts in the model.
        """
        return f"Model(name={self.name}, prompts={list(self.prompts.keys())})"

class ModelPrompt:
    def __init__(self, main_folder_path='uat_eval/common/prompt_manager/prompts'):
        self.main_folder_path = main_folder_path
        self.models = {}
        self.update_models()

    def add_model(self, name):
        """Add a model to the collection.

        Args:
            name (str): The name of the model to add.
        """
        if name in self.models:
            warnings.warn(f"Model '{name}' already exists.")
        else:
            self.models[name] = Model(name, self.main_folder_path)

    def get_model(self, name):
        """Get a model from the collection.

        Args:
            name (str): The name of the model to get.

        Returns:
            Model: The model with the given name, or 'Model not found' if the model does not exist.
        """
        if name not in self.models:
            model_folder_path = os.path.join(self.main_folder_path, name)
            if os.path.exists(model_folder_path):
                self.models[name] = Model(name, self.main_folder_path)
            else:
                return 'Model not found'
        return self.models[name]

    def get_models(self):
        """Get a list of the names of the models in the collection.

        Returns:
            list: A list of the names of the models in the collection.
        """
        return list(self.models.keys())

    def delete_model(self, name):
        """Delete a model from the collection.

        Args:
            name (str): The name of the model to delete.
        """
        model = self.get_model(name)
        if model != 'Model not found':
            if os.path.exists(model.folder_path):
                shutil.rmtree(model.folder_path)
                del self.models[name]
            else:
                warnings.warn(f"Model '{name}' does not exist in the folder.")
        else:
            warnings.warn(f"Model '{name}' does not exist.")

    def update_models(self):
        """Update the models in the collection based on the folders in the main folder path."""
        for folder_name in os.listdir(self.main_folder_path):
            folder_path = os.path.join(self.main_folder_path, folder_name)
            if os.path.isdir(folder_path) and folder_name not in self.models:
                self.models[folder_name] = Model(folder_name, self.main_folder_path)

    def __str__(self):
        """Return a string representation of the collection.

        Returns:
            str: A string representation of the collection, including the names of the models in the collection.
        """
        return f"ModelPrompt(models={list(self.models.keys())})"
