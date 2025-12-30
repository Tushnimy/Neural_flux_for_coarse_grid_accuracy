import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="hybrid_fvm_pinn")
class NeuralFlux(tf.keras.Model):
    """
    Neural numerical flux for Burgers equation.
    Learns dissipation term phi_theta(uL, uR, log_mu).
    """
    def __init__(self, hidden_sizes=(64, 64), activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

        self.net = tf.keras.Sequential(name="phi_net")
        for h in self.hidden_sizes:
            self.net.add(tf.keras.layers.Dense(h, activation=self.activation))
        self.net.add(tf.keras.layers.Dense(1, activation=None))

    @staticmethod
    def physical_flux(u):
        # Burgers: f(u) = 1/2 u^2
        return 0.5 * tf.square(u)

    def call(self, inputs):
        # inputs: [uL, uR, log_mu]
        uL = inputs[:, 0:1]
        uR = inputs[:, 1:2]
        log_mu = inputs[:, 2:3]

        phi = self.net(tf.concat([uL, uR, log_mu], axis=1))

        fL = self.physical_flux(uL)
        fR = self.physical_flux(uR)
        f_hat = 0.5 * (fL + fR) + (uR - uL) * phi
        return f_hat

    def get_config(self):
        return {"hidden_sizes": self.hidden_sizes, "activation": self.activation}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
