# Classical Composer
Classical Composer model used for generating classical pieces of music in MIDI format.
Created as 2nd project for AI course at elfak.

Structure
=========
It consists of 3 parts (modules):
* Preprocessing of MIDI files to be compatible as inputs for model (`preprocessing.py`)
* Defining model architecture and training process (`model.py`)
* Generating MIDI files as model's predictions (`generator.py`) 



Install
=======
Create new virtual environment and activate it.

`
$ python -m venv composer_env
$ source composer_env/bin/activate
`

Then install all dependencies from `requirements.txt` file.

`
(composer_env)$ pip install -r requirements.txt
`
