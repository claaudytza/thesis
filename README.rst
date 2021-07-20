You can perform a minimal install of ``gym`` with:

.. code:: shell

    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .

If you prefer, you can do a minimal install of the packaged version directly from PyPI:

.. code:: shell

    pip install gym

You'll be able to run a few environments right away:

- algorithmic
- toy_text
- classic_control (you'll need ``pyglet`` to render though)


You can also `run gym on gitpod.io <https://gitpod.io/#https://github.com/openai/gym/blob/master/examples/agents/cem.py>`_ to play with the examples online.  
In the preview window you can click on the mp4 file you want to view. If you want to view another mp4 file, just press the back button and click on another mp4 file. 

Installing everything
---------------------

To install the full set of environments, you'll need to have some system
packages installed. We'll build out the list here over time; please let us know
what you end up installing on your platform. Also, take a look at the docker files (py.Dockerfile) to
see the composition of our CI-tested images.

On Ubuntu 16.04 and 18.04:

.. code:: shell
    
    apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig

MuJoCo has a proprietary dependency we can't set up for you. Follow
the
`instructions <https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key>`_
in the ``mujoco-py`` package for help. Note that we currently do not support MuJoCo 2.0 and above, so you will need to install a version of mujoco-py which is built
for a lower version of MuJoCo like MuJoCo 1.5 (example - ``mujoco-py-1.50.1.0``).
As an alternative to ``mujoco-py``, consider `PyBullet <https://github.com/openai/gym/blob/master/docs/environments.md#pybullet-robotics-environments>`_ which uses the open source Bullet physics engine and has no license requirement.

Once you're ready to install everything, run ``pip install -e '.[all]'`` (or ``pip install 'gym[all]'``).

Pip version
-----------

To run ``pip install -e '.[all]'``, you'll need a semi-recent pip.
Please make sure your pip is at least at version ``1.5.0``. You can
upgrade using the following: ``pip install --ignore-installed
pip``. Alternatively, you can open `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_ and
install the dependencies by hand.

WaterWorld
-----------

Clone the github: https://github.com/claaudytza/gym-waterworld and then install the package with pip install -e gym-waterworld. You can now create an instance of the environment with gym.make('gym_waterworld:waterworld-v0')

Rendering on a server
---------------------

If you're trying to render video on a server, you'll need to connect a
fake display. The easiest way to do this is by running under
``xvfb-run`` (on Ubuntu, install the ``xvfb`` package):

.. code:: shell

     xvfb-run -s "-screen 0 1400x900x24" bash

Installing dependencies for specific environments
-------------------------------------------------

If you'd like to install the dependencies for only specific
environments, see `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_. We
maintain the lists of dependencies on a per-environment group basis.

Environments
============

See `List of Environments <docs/environments.md>`_ and the `gym site <http://gym.openai.com/envs/>`_.

For information on creating your own environments, see `Creating your own Environments <docs/creating-environments.md>`_.

Examples
========

See the ``examples`` directory.

- Run `examples/agents/random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ to run a simple random agent.
- Run `examples/agents/cem.py <https://github.com/openai/gym/blob/master/examples/agents/cem.py>`_ to run an actual learning agent (using the cross-entropy method).
- Run `examples/scripts/list_envs <https://github.com/openai/gym/blob/master/examples/scripts/list_envs>`_ to generate a list of all environments.

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

    pytest


.. _See What's New section below:

Resources
=========

-  `OpenAI.com`_
-  `Gym.OpenAI.com`_
-  `Gym Docs`_
-  `Gym Environments`_
-  `OpenAI Twitter`_
-  `OpenAI YouTube`_

.. _OpenAI.com: https://openai.com/
.. _Gym.OpenAI.com: http://gym.openai.com/
.. _Gym Docs: http://gym.openai.com/docs/
.. _Gym Environments: http://gym.openai.com/envs/
.. _OpenAI Twitter: https://twitter.com/openai
.. _OpenAI YouTube: https://www.youtube.com/channel/UCXZCJLdBC09xxGZ6gcdrc6A
