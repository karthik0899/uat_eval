# Prompt Manager Documentation

This document provides an overview of the Prompt Manager package, which allows you to manage models and prompts for various language models. The package includes two classes: `Model` and `ModelPrompt`. The `Model` class represents a model with a name and a list of prompts, while the `ModelPrompt` class represents a collection of models.

## Folder Structure

The folder structure for the Prompt Manager package is as follows:

```
common/
└── prompt_manager/
    ├── __init__.py
    ├── PromptManager.py
    └── prompts/
        └── <model_name>/
            └── <prompt_name>/
                └── system.txt
```

* The `common/prompt_manager` folder is the main folder for the `prompt_manager` package.
* The `__init__.py` file tells Python that the `prompt_manager` folder should be treated as a package.
* The `PromptManager.py` file contains the `Model` and `ModelPrompt` classes that we defined earlier.
* The `prompts` folder is where all the model and prompt folders are located. Each model has its own folder, and each prompt has its own folder within the model folder. The `system.txt` file contains the content of the prompt.

## Code Documentation

### `Model` class

The `Model` class represents a model with a name and a list of prompts. The `Model` class has the following methods:

* `__init__(self, name, main_folder_path='common/prompt_manager/prompts')`: Initializes a new instance of the `Model` class with the given name and main folder path. Creates a folder for the model if it doesn't already exist. Updates the prompts in the model based on the folders in the model folder path.
* `add_prompt(self, name, content)`: Adds a prompt to the model with the given name and content. Creates a folder for the prompt and saves the content to a `system.txt` file in that folder.
* `get_prompt(self, name, context)`: Gets the content of a prompt with the given name and context. Replaces the `{context}` placeholder in the prompt content with the given context.
* `get_prompts(self)`: Gets a list of the names of the prompts in the model.
* `delete_prompt(self, name)`: Deletes a prompt with the given name from the model.
* `update_prompts(self)`: Updates the prompts in the model based on the folders in the model folder path.
* `__str__(self)`: Returns a string representation of the model, including the name of the model and the names of the prompts in the model.

### `ModelPrompt` class

The `ModelPrompt` class represents a collection of models. The `ModelPrompt` class has the following methods:

* `__init__(self, main_folder_path='common/prompt_manager/prompts')`: Initializes a new instance of the `ModelPrompt` class with the given main folder path. Updates the models in the collection based on the folders in the main folder path.
* `add_model(self, name)`: Adds a model to the collection with the given name.
* `get_model(self, name)`: Gets a model from the collection with the given name. If the model does not exist in the collection, checks if the model folder exists in the main folder path. If the model folder exists, creates a new `Model` instance for the model and adds it to the collection. If the model folder does not exist, returns 'Model not found'.
* `get_models(self)`: Gets a list of the names of the models in the collection.
* `delete_model(self, name)`: Deletes a model with the given name from the collection.
* `update_models(self)`: Updates the models in the collection based on the folders in the main folder path.
* `__str__(self)`: Returns a string representation of the collection, including the names of the models in the collection.

### Adding Models and Prompts Manually

You can add models and prompts manually to the folder by creating the necessary folders and files. Here's how to do it:

1. Create a new folder for the model inside the `prompts` folder. The name of the folder should be the name of the model.
2. Inside the model folder, create a new folder for each prompt. The name of the folder should be the name of the prompt.
3. Inside each prompt folder, create a new file called `system.txt`. This file should contain the content of the prompt.

For example, to add a model called `new_model` with a prompt called `new_prompt`, you would create the following folder structure:

```
common/
└── prompt_manager/
    └── prompts/
        └── new_model/
            └── new_prompt/
                └── system.txt
```

The `system.txt` file should contain the content of the prompt. For example:

```
New prompt content: {context}
```

After adding the model and prompt manually, you can use the `ModelPrompt` class to retrieve the prompt for the model and context. Here's an example:

```python
from common.prompt_manager import PromptManager

# Create a new instance of the ModelPrompt class
model_prompt = PromptManager.ModelPrompt()

# The models in the collection are automatically updated based on the folders in the main folder path
print(model_prompt.get_models())

# Get the new model from the collection
model = model_prompt.get_model('new_model')

# The new model is now in the collection
print(model_prompt.get_models())

# Get the prompt from the new model
prompt = model.get_prompt('new_prompt', 'actual context')
print(prompt)
```

In this example, we import the `PromptManager` class from the `common.prompt_manager` package using the `from ... import` syntax. We then use the `ModelPrompt` class to get the new model from the collection, and retrieve the prompt for the model and context.

### Usage

Here's an example of how to use the `PromptManager` class from the `prompt_manager` package to add models and prompts, and retrieve prompts for a given model and context:

```python
from common.prompt_manager import PromptManager

# Create a new instance of the ModelPrompt class
model_prompt = PromptManager.ModelPrompt()

# The models in the collection are automatically updated based on the folders in the main folder path
print(model_prompt.get_models())

# Add a new model to the collection
model_prompt.add_model('new_model')

# The new model is automatically added to the collection
print(model_prompt.get_models())

# Get the new model from the collection
model = model_prompt.get_model('new_model')

# Add a prompt to the new model
model.add_prompt('new_prompt', 'New prompt content: {context}')

# Get the prompt from the new model
prompt = model.get_prompt('new_prompt', 'actual context')
print(prompt)

# Delete the prompt from the new model
model.delete_prompt('new_prompt')

# The prompt is no longer in the new model
print(model.get_prompts())

# Delete the new model from the collection
model_prompt.delete_model('new_model')

# The new model is no longer in the collection
print(model_prompt.get_models())
```