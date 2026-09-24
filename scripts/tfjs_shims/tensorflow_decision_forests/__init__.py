"""Empty stand-in for `tensorflow_decision_forests`.

tensorflowjs 4.22 imports TF-DF at module load only to register decision-forest custom ops. TF-DF has no
Windows wheels and pins an exact TensorFlow version, so the conversion venv skips it. This model contains
no decision-forest ops, so the stub changes nothing about the conversion.
"""
