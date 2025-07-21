from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Crear entorno
env = EscobaEnv()

# Verifica que el entorno cumpla con la API de Gym
check_env(env)

# Entrenar el agente DQN
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=5000, exploration_fraction=0.2)
model.learn(total_timesteps=100)

# Guardar el modelo entrenado
model.save("escoba_dqn_model")
print("Entrenamiento completo y modelo guardado.")
