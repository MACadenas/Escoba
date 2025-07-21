from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from escoba_env import EscobaEnv
import os
import json

COUNTER_PATH = "escoba_training_counter.json"

def load_training_counter():
    if os.path.exists(COUNTER_PATH):
        with open(COUNTER_PATH, "r") as f:
            return json.load(f).get("total_steps", 0)
    return 0

def save_training_counter(total_steps):
    with open(COUNTER_PATH, "w") as f:
        json.dump({"total_steps": total_steps}, f)


def train_agent(total_new_steps=1000):
    env = EscobaEnv()

    # Cargar modelo si existe
    if os.path.exists("escoba_dqn_model.zip"):
        print("Cargando modelo existente...")
        model = DQN.load("escoba_dqn_model", env=env)
    else:
        print("Creando nuevo modelo...")
        model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=5000, exploration_fraction=0.2)

    # Cargar replay buffer si existe
    if os.path.exists("escoba_dqn_model_buffer.pkl"):
        model.load_replay_buffer("escoba_dqn_model_buffer")
        print("Replay buffer cargado.")

    # Cargar contador de steps previos
    previous_steps = load_training_counter()
    print(f"Steps entrenados previamente: {previous_steps}")

    # Entrenar
    model.learn(total_timesteps=total_new_steps)

    # Guardar modelo y buffer
    model.save("escoba_dqn_model")
    model.save_replay_buffer("escoba_dqn_model_buffer")

    # Guardar nuevo contador
    total_steps = previous_steps + total_new_steps
    save_training_counter(total_steps)
    print(f"Entrenamiento completo. Total acumulado de steps: {total_steps}")

if __name__ == "__main__":
    train_agent()
