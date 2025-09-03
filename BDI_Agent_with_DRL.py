import numpy as np
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import os
import math
from typing import Tuple, Dict, List, Optional
import heapq
import matplotlib.pyplot as plt

# Configuración de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PrioritizedReplayBuffer:
    """Buffer de experiencia con priorización para mejorar el aprendizaje."""

    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, priority=None):
        """Añade experiencia al buffer con prioridad."""
        if priority is None:
            priority = max(self.priorities) if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """Muestrea batch con priorización."""
        if self.size == 0:
            return None

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Actualiza prioridades basado en TD-error."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class SmartAStarGuidance:
    """Guía inteligente usando A* con adaptaciones para el problema."""

    def __init__(self, grid_resolution: float = 2.0):
        self.grid_resolution = grid_resolution
        self.cache = {}  # Cache de rutas calculadas

    def heuristic(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Heurística euclidiana mejorada."""
        distance = np.linalg.norm(pos1 - pos2)
        # Penalizar movimientos en Z para mantener estabilidad
        z_penalty = abs(pos1[2] - pos2[2]) * 0.5
        return distance + z_penalty

    def get_best_direction(self, current_pos: np.ndarray, target_pos: np.ndarray,
                          previous_directions: List[np.ndarray] = None) -> np.ndarray:
        """Obtiene la mejor dirección hacia el objetivo."""
        direct_vector = target_pos - current_pos
        distance = np.linalg.norm(direct_vector)

        if distance < 1e-6:
            return np.zeros(3)

        # Dirección normalizada
        direction = direct_vector / distance

        # Suavizar cambios bruscos si hay historial
        if previous_directions and len(previous_directions) > 0:
            avg_previous = np.mean(previous_directions[-3:], axis=0)
            # Mezcla entre dirección óptima y suavizada
            smoothing_factor = min(0.3, 10.0 / max(distance, 1.0))
            direction = (1 - smoothing_factor) * direction + smoothing_factor * avg_previous
            direction = direction / (np.linalg.norm(direction) + 1e-8)

        return direction

    def get_action_from_direction(self, direction: np.ndarray, action_vectors: Dict) -> int:
        """Convierte dirección en acción discreta."""
        if np.linalg.norm(direction) < 1e-6:
            return 0

        best_action = 0
        best_similarity = -2.0

        for action_id, action_vector in action_vectors.items():
            similarity = np.dot(direction, action_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = action_id

        return best_action

class StabilizedDQNAgent:
    """Agente DQN completamente rediseñado para estabilidad y convergencia."""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        # Parámetros optimizados
        self.gamma = 0.995  # Discount factor alto para planificación a largo plazo
        self.epsilon = 0.95  # Exploración inicial alta
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9995  # Decaimiento más gradual
        self.learning_rate = 0.0001  # Learning rate muy bajo para estabilidad
        self.batch_size = 64
        self.target_update_freq = 200  # Menos frecuente para estabilidad

        # Buffers y contadores
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.update_counter = 0
        self.training_step = 0

        # Redes neuronales
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Guía heurística
        self.guidance = SmartAStarGuidance()
        self.direction_history = deque(maxlen=5)

        # Métricas de entrenamiento
        self.loss_history = deque(maxlen=1000)
        self.q_value_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=100)

        # Parámetros adaptativos
        self.success_streak = 0
        self.failure_streak = 0

    def _build_model(self) -> tf.keras.Model:
        """Red neuronal optimizada con regularización."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),

            # Capas densas con normalización y dropout
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            # Capa de salida
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])

        # Optimizador con cliping de gradientes
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0  # Gradient clipping
        )

        model.compile(
            optimizer=optimizer,
            loss='huber',  # Menos sensible a outliers
            metrics=['mae']
        )

        return model

    def update_target_model(self):
        """Soft update de la red objetivo."""
        tau = 0.005  # Muy gradual
        target_weights = self.target_model.get_weights()
        model_weights = self.model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def adaptive_epsilon_update(self, episode_success: bool):
        """Actualización adaptativa de epsilon basada en rendimiento."""
        if episode_success:
            self.success_streak += 1
            self.failure_streak = 0
            # Reducir exploración si hay éxitos consecutivos
            if self.success_streak >= 3:
                self.epsilon *= 0.98
        else:
            self.failure_streak += 1
            self.success_streak = 0
            # Aumentar exploración si hay muchos fallos
            if self.failure_streak >= 5:
                self.epsilon = min(0.4, self.epsilon * 1.1)

        # Aplicar límites
        self.epsilon = max(self.epsilon_min, min(0.95, self.epsilon))

    def choose_action(self, state: np.ndarray, current_pos: np.ndarray = None,
                     target_pos: np.ndarray = None, training: bool = True) -> int:
        """Selección de acción con guía inteligente."""

        # Definir vectores de acción
        action_vectors = {
            0: np.array([1, 0, 0]),    # +X
            1: np.array([-1, 0, 0]),   # -X
            2: np.array([0, 1, 0]),    # +Y
            3: np.array([0, -1, 0]),   # -Y
            4: np.array([0, 0, 1]),    # +Z
            5: np.array([0, 0, -1]),   # -Z
            6: np.array([0.707, 0.707, 0]),    # Diagonal XY
            7: np.array([0, 0.707, 0.707]),    # Diagonal YZ
            8: np.array([0.707, 0, 0.707])     # Diagonal XZ
        }

        if training and np.random.rand() <= self.epsilon:
            # Exploración guiada por heurística
            if current_pos is not None and target_pos is not None:
                optimal_direction = self.guidance.get_best_direction(
                    current_pos, target_pos, list(self.direction_history)
                )

                # 70% probabilidad de seguir guía, 30% exploración pura
                if np.random.rand() < 0.7:
                    action = self.guidance.get_action_from_direction(optimal_direction, action_vectors)
                    self.direction_history.append(optimal_direction)
                    return action

            # Exploración pura
            return random.randrange(self.action_size)

        # Explotación
        if len(state.shape) == 1:
            state = np.reshape(state, [1, -1])

        q_values = self.model.predict(state, verbose=0)
        self.q_value_history.append(np.mean(q_values))

        return np.argmax(q_values[0])

    def store_transition(self, state, action, reward, next_state, done):
        """Almacena transición con priorización."""
        # Calcular TD-error como prioridad
        if len(state.shape) == 1:
            state = np.reshape(state, [1, -1])
            next_state = np.reshape(next_state, [1, -1])

        current_q = self.model.predict(state, verbose=0)[0][action]

        if done:
            target_q = reward
        else:
            next_q_values = self.target_model.predict(next_state, verbose=0)
            target_q = reward + self.gamma * np.max(next_q_values[0])

        priority = abs(target_q - current_q) + 0.01  # Pequeño offset para evitar 0

        self.memory.add(state[0], action, reward, next_state[0], done, priority)

    def learn(self):
        """Entrenamiento mejorado con estabilización."""
        if self.memory.size < self.batch_size * 2:
            return

        # Muestreo con priorización
        batch_data = self.memory.sample(self.batch_size, beta=0.4)
        if batch_data is None:
            return

        experiences, indices, weights = batch_data

        # Desempaquear experiencias
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])

        # Double DQN
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = next_q_values[np.arange(self.batch_size), next_actions]

        # Calcular targets
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Q-values actuales
        current_q_values = self.model.predict(states, verbose=0)

        # Calcular TD-errors para actualizar prioridades
        td_errors = []
        for i in range(self.batch_size):
            old_q = current_q_values[i][actions[i]]
            current_q_values[i][actions[i]] = targets[i]
            td_errors.append(abs(targets[i] - old_q))

        # Actualizar prioridades
        self.memory.update_priorities(indices, td_errors)

        # Entrenar con importance sampling weights
        sample_weights = np.array(weights)
        history = self.model.fit(
            states, current_q_values,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            sample_weight=sample_weights
        )

        self.loss_history.append(history.history['loss'][0])
        self.training_step += 1

        # Decaimiento de epsilon más gradual
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Actualización de red objetivo
        if self.training_step % self.target_update_freq == 0:
            self.update_target_model()

class OptimizedAirCombatEnv(gym.Env):
    """Entorno optimizado con sistema de recompensas balanceado."""

    def __init__(self, target_pos: List[float], max_steps: int = 150):
        super().__init__()

        self.max_steps = max_steps
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.success_distance = 2.0

        # Estados del sistema
        self.current_step = 0
        self.uav_pos = np.zeros(3, dtype=np.float32)
        self.previous_distance = 0.0
        self.initial_distance = 0.0
        self.best_distance = float('inf')

        # Historial para análisis
        self.distance_history = []
        self.position_history = []
        self.action_history = []

        # Espacio de observación normalizado
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(9)

        # Parámetros de movimiento optimizados
        self.max_speed = 2.5
        self.movements = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
            6: np.array([0.707, 0.707, 0]),
            7: np.array([0, 0.707, 0.707]),
            8: np.array([0.707, 0, 0.707])
        }

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normaliza posición al rango [-1, 1]."""
        return np.clip(pos / 100.0, -1, 1)

    def _normalize_distance(self, distance: float) -> float:
        """Normaliza distancia al rango [0, 1]."""
        return min(distance / 200.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        """Observación normalizada y rica en información."""
        current_distance = np.linalg.norm(self.uav_pos - self.target_pos)

        # Vector direccional normalizado
        direction_vector = self.target_pos - self.uav_pos
        if np.linalg.norm(direction_vector) > 0:
            direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Progreso relativo
        progress = 1.0 - (current_distance / max(self.initial_distance, 1.0))
        progress = max(0, min(progress, 1.0))

        # Velocidad de aproximación
        distance_change = self.previous_distance - current_distance
        velocity_factor = np.tanh(distance_change)  # Normalizado entre -1 y 1

        return np.concatenate([
            self._normalize_position(self.uav_pos),          # Posición UAV [3]
            self._normalize_position(self.target_pos),       # Posición objetivo [3]
            [self._normalize_distance(current_distance)],    # Distancia actual [1]
            [self._normalize_distance(self.initial_distance)], # Distancia inicial [1]
            direction_vector,                                 # Dirección normalizada [3]
            [progress]                                        # Progreso [1]
        ])

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset optimizado con posiciones iniciales estratégicas."""
        super().reset(seed=seed)

        # Posición inicial controlada
        distance_range = [30, 80]
        initial_distance = self.np_random.uniform(distance_range[0], distance_range[1])

        # Ángulos aleatorios para diversidad
        azimuth = self.np_random.uniform(0, 2 * np.pi)
        elevation = self.np_random.uniform(-np.pi/6, np.pi/6)

        # Calcular posición inicial
        self.uav_pos = self.target_pos + initial_distance * np.array([
            np.cos(azimuth) * np.cos(elevation),
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation)
        ])

        # Inicializar métricas
        self.current_step = 0
        self.initial_distance = np.linalg.norm(self.uav_pos - self.target_pos)
        self.previous_distance = self.initial_distance
        self.best_distance = self.initial_distance

        # Limpiar historiales
        self.distance_history = [self.initial_distance]
        self.position_history = [self.uav_pos.copy()]
        self.action_history = []

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Paso de simulación con recompensas balanceadas."""
        self.current_step += 1
        self.action_history.append(action)

        # Aplicar movimiento
        movement = self.movements[action] * self.max_speed
        new_position = self.uav_pos + movement

        # Límites suaves para evitar salirse del espacio
        new_position = np.clip(new_position, -120, 120)
        self.uav_pos = new_position

        # Calcular distancia y métricas
        current_distance = np.linalg.norm(self.uav_pos - self.target_pos)
        self.distance_history.append(current_distance)
        self.position_history.append(self.uav_pos.copy())

        # Actualizar mejor distancia
        if current_distance < self.best_distance:
            self.best_distance = current_distance

        # Sistema de recompensas balanceado
        reward = self._calculate_balanced_reward(current_distance)

        # Actualizar distancia previa
        self.previous_distance = current_distance

        # Condiciones de terminación
        success = current_distance < self.success_distance
        timeout = self.current_step >= self.max_steps
        out_of_bounds = np.any(np.abs(self.uav_pos) > 150)

        done = success or out_of_bounds
        truncated = timeout and not done

        # Recompensas/penalizaciones finales
        if success:
            reward += 1000 + (self.max_steps - self.current_step) * 5
        elif out_of_bounds:
            reward -= 500
        elif truncated:
            reward -= 100

        return self._get_obs(), reward, done, truncated, self._get_info()

    def _calculate_balanced_reward(self, current_distance: float) -> float:
        """Sistema de recompensas completamente balanceado."""
        reward = 0.0

        # 1. Recompensa por acercarse (componente principal)
        distance_improvement = self.previous_distance - current_distance

        # Escalado no lineal que favorece estar cerca
        proximity_bonus = 1.0 / (1.0 + current_distance / 10.0)
        reward += distance_improvement * 50.0 * (1.0 + proximity_bonus)

        # 2. Recompensa por mantener distancia mínima
        if current_distance < self.best_distance:
            reward += 20.0

        # 3. Recompensa por progreso hacia el objetivo
        progress = (self.initial_distance - current_distance) / self.initial_distance
        if progress > 0:
            reward += progress * 10.0

        # 4. Penalización por alejarse demasiado
        if current_distance > self.initial_distance * 1.2:
            reward -= 30.0

        # 5. Penalización suave por tiempo
        reward -= 0.1

        # 6. Bonus por proximidad extrema
        if current_distance < 5.0:
            reward += 50.0 / max(current_distance, 0.1)

        return reward

    def _get_info(self) -> Dict:
        """Información detallada del estado."""
        current_distance = np.linalg.norm(self.uav_pos - self.target_pos)
        return {
            "distance": current_distance,
            "best_distance": self.best_distance,
            "progress": 1.0 - (current_distance / self.initial_distance),
            "success": current_distance < self.success_distance,
            "steps": self.current_step,
            "position": self.uav_pos.copy(),
            "efficiency": self.best_distance / self.initial_distance
        }

def train_stabilized_agent(env, agent, num_episodes: int = 300):
    """Sistema de entrenamiento optimizado con monitoreo avanzado."""

    print("🚀 Iniciando entrenamiento optimizado...")
    print("=" * 70)

    # Métricas de entrenamiento
    episode_rewards = []
    success_episodes = []
    episode_lengths = []
    final_distances = []
    best_distances = []

    # Métricas de convergencia
    recent_success_rate = 0
    best_success_rate = 0
    episodes_since_improvement = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < env.max_steps:
            # Selección de acción con guía
            action = agent.choose_action(
                state,
                env.uav_pos,
                env.target_pos,
                training=True
            )

            next_state, reward, done, truncated, info = env.step(action)

            # Almacenar experiencia
            agent.store_transition(state, action, reward, next_state, done or truncated)

            # Aprender
            if step_count % 4 == 0:  # Aprender cada 4 pasos
                agent.learn()

            state = next_state
            total_reward += reward
            step_count += 1

            if done or truncated:
                break

        # Registrar métricas del episodio
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        final_distances.append(info['distance'])
        best_distances.append(info['best_distance'])

        is_success = info['success']
        success_episodes.append(1 if is_success else 0)

        # Actualización adaptativa de epsilon
        agent.adaptive_epsilon_update(is_success)

        # Calcular métricas móviles
        window_size = min(20, episode)
        recent_success_rate = np.mean(success_episodes[-window_size:])
        recent_avg_reward = np.mean(episode_rewards[-window_size:])
        recent_avg_distance = np.mean(final_distances[-window_size:])
        recent_best_distance = np.mean(best_distances[-window_size:])

        # Detectar mejora
        if recent_success_rate > best_success_rate:
            best_success_rate = recent_success_rate
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1

        # Log detallado cada 10 episodios
        if episode % 10 == 0:
            status = "✅ ÉXITO" if is_success else "❌ FALLO"

            print(f"Episodio {episode:3d}/{num_episodes} | {status}")
            print(f"  Pasos: {step_count:3d} | Recompensa: {total_reward:8.1f} | "
                  f"Dist. Final: {info['distance']:6.2f}m | Mejor: {info['best_distance']:6.2f}m")
            print(f"  Métricas ({window_size} episodios):")
            print(f"    Tasa Éxito: {recent_success_rate:6.1%} | "
                  f"Recompensa Media: {recent_avg_reward:8.1f}")
            print(f"    Dist. Final Media: {recent_avg_distance:6.2f}m | "
                  f"Mejor Dist. Media: {recent_best_distance:6.2f}m")
            print(f"  Parámetros Agente:")
            print(f"    Epsilon: {agent.epsilon:.4f} | "
                  f"Pasos Entren.: {agent.training_step}")
            print(f"    Loss Media: {np.mean(list(agent.loss_history)[-100:]):.4f}")
            print("-" * 70)

        # Early stopping si hay convergencia
        if episode > 100 and recent_success_rate >= 0.9 and episodes_since_improvement > 50:
            print(f"\n🎯 Convergencia detectada en episodio {episode}")
            print(f"   Tasa de éxito sostenida: {recent_success_rate:.1%}")
            break

    # Resumen final
    total_successes = sum(success_episodes)
    final_success_rate = total_successes / len(success_episodes)

    print("\n" + "=" * 70)
    print("🏆 RESUMEN FINAL DEL ENTRENAMIENTO")
    print("=" * 70)
    print(f"Episodios completados: {episode}/{num_episodes}")
    print(f"Éxitos totales: {total_successes}")
    print(f"Tasa de éxito final: {final_success_rate:.1%}")
    print(f"Mejor tasa de éxito: {best_success_rate:.1%}")
    print(f"Recompensa promedio (últimos 20): {np.mean(episode_rewards[-20:]):.1f}")
    print(f"Distancia final promedio: {np.mean(final_distances[-20:]):.2f}m")
    print(f"Mejor distancia promedio: {np.mean(best_distances[-20:]):.2f}m")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print("=" * 70)

    return {
        'rewards': episode_rewards,
        'success_episodes': success_episodes,
        'final_distances': final_distances,
        'best_distances': best_distances,
        'success_rate': final_success_rate,
        'episodes_completed': episode
    }

def create_training_plots(results: Dict):
    """Crea gráficos de análisis del entrenamiento."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis del Entrenamiento DQN Optimizado', fontsize=16, fontweight='bold')

    episodes = range(1, len(results['rewards']) + 1)

    # 1. Evolución de recompensas
    axes[0,0].plot(episodes, results['rewards'], alpha=0.6, color='blue', linewidth=0.8)

    # Media móvil de recompensas
    window = 20
    if len(results['rewards']) >= window:
        moving_avg = [np.mean(results['rewards'][max(0, i-window):i+1])
                     for i in range(len(results['rewards']))]
        axes[0,0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'Media móvil ({window})')

    axes[0,0].set_xlabel('Episodio')
    axes[0,0].set_ylabel('Recompensa')
    axes[0,0].set_title('Evolución de Recompensas')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Tasa de éxito
    success_rate = []
    window_success = 20
    for i in range(len(results['success_episodes'])):
        start_idx = max(0, i - window_success + 1)
        rate = np.mean(results['success_episodes'][start_idx:i+1])
        success_rate.append(rate * 100)

    axes[0,1].plot(episodes, success_rate, color='green', linewidth=2)
    axes[0,1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Objetivo 80%')
    axes[0,1].set_xlabel('Episodio')
    axes[0,1].set_ylabel('Tasa de Éxito (%)')
    axes[0,1].set_title('Evolución de Tasa de Éxito')
    axes[0,1].set_ylim(0, 100)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Distancias finales
    axes[1,0].scatter(episodes, results['final_distances'], alpha=0.6, s=20, color='purple')
    axes[1,0].axhline(y=2.0, color='red', linestyle='-', linewidth=2, label='Umbral de éxito (2m)')

    # Media móvil de distancias
    if len(results['final_distances']) >= window:
        dist_moving_avg = [np.mean(results['final_distances'][max(0, i-window):i+1])
                          for i in range(len(results['final_distances']))]
        axes[1,0].plot(episodes, dist_moving_avg, color='orange', linewidth=2,
                      label=f'Media móvil ({window})')

    axes[1,0].set_xlabel('Episodio')
    axes[1,0].set_ylabel('Distancia Final (m)')
    axes[1,0].set_title('Evolución de Distancia Final')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Comparación de mejores distancias
    axes[1,1].plot(episodes, results['best_distances'], color='darkgreen',
                  linewidth=1.5, label='Mejor distancia por episodio')
    axes[1,1].axhline(y=2.0, color='red', linestyle='-', linewidth=2,
                     label='Umbral de éxito (2m)')

    # Percentiles
    if len(results['best_distances']) >= 10:
        p25 = np.percentile(results['best_distances'], 25)
        p75 = np.percentile(results['best_distances'], 75)
        axes[1,1].axhline(y=p25, color='blue', linestyle=':', alpha=0.7, label='Percentil 25')
        axes[1,1].axhline(y=p75, color='blue', linestyle=':', alpha=0.7, label='Percentil 75')

    axes[1,1].set_xlabel('Episodio')
    axes[1,1].set_ylabel('Mejor Distancia (m)')
    axes[1,1].set_title('Evolución de Mejor Distancia Alcanzada')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Guardar gráfico
    filename = 'training_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráficos guardados como {filename}")

    plt.show()

def run_final_test(env, agent, num_test_episodes: int = 5):
    """Prueba final del agente entrenado."""
    print(f"\n🧪 PRUEBA FINAL DEL AGENTE ({num_test_episodes} episodios)")
    print("=" * 60)

    # Desactivar exploración para prueba
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    test_results = []

    for episode in range(1, num_test_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        trajectory = [env.uav_pos.copy()]

        while not done and step_count < env.max_steps:
            action = agent.choose_action(state, env.uav_pos, env.target_pos, training=False)
            next_state, reward, done, truncated, info = env.step(action)

            trajectory.append(env.uav_pos.copy())
            state = next_state
            total_reward += reward
            step_count += 1

            if done or truncated:
                break

        # Registrar resultado
        result = {
            'episode': episode,
            'success': info['success'],
            'steps': step_count,
            'final_distance': info['distance'],
            'best_distance': info['best_distance'],
            'total_reward': total_reward,
            'efficiency': info['efficiency'],
            'trajectory': trajectory
        }
        test_results.append(result)

        # Mostrar resultado
        status = "✅ ÉXITO" if info['success'] else "❌ FALLO"
        print(f"Prueba {episode}: {status}")
        print(f"  Pasos: {step_count:3d} | Recompensa: {total_reward:8.1f}")
        print(f"  Distancia final: {info['distance']:6.2f}m | Mejor: {info['best_distance']:6.2f}m")
        print(f"  Eficiencia: {info['efficiency']:6.3f}")
        print("-" * 40)

    # Estadísticas finales
    successes = sum(1 for r in test_results if r['success'])
    success_rate = successes / num_test_episodes
    avg_steps = np.mean([r['steps'] for r in test_results])
    avg_final_distance = np.mean([r['final_distance'] for r in test_results])
    avg_best_distance = np.mean([r['best_distance'] for r in test_results])

    print("\n" + "=" * 60)
    print("📋 RESULTADOS DE LA PRUEBA FINAL")
    print("=" * 60)
    print(f"Tasa de éxito: {success_rate:.1%} ({successes}/{num_test_episodes})")
    print(f"Pasos promedio: {avg_steps:.1f}")
    print(f"Distancia final promedio: {avg_final_distance:.2f}m")
    print(f"Mejor distancia promedio: {avg_best_distance:.2f}m")

    if successes > 0:
        successful_tests = [r for r in test_results if r['success']]
        avg_success_steps = np.mean([r['steps'] for r in successful_tests])
        avg_success_efficiency = np.mean([r['efficiency'] for r in successful_tests])

        print(f"\nMÉTRICAS DE EPISODIOS EXITOSOS:")
        print(f"Pasos promedio en éxitos: {avg_success_steps:.1f}")
        print(f"Eficiencia promedio en éxitos: {avg_success_efficiency:.3f}")

    print("=" * 60)

    # Restaurar epsilon original
    agent.epsilon = original_epsilon

    return test_results

# Función principal mejorada
def main():
    """Función principal con configuración optimizada."""
    print("🚁 Sistema de Interceptación UAV - Entrenamiento Optimizado")
    print("=" * 70)

    # Configuración del sistema
    target_position = [25, 15, 10]
    max_steps = 150

    # Crear entorno optimizado
    env = OptimizedAirCombatEnv(target_pos=target_position, max_steps=max_steps)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print(f"Configuración del sistema:")
    print(f"- Posición objetivo: {target_position}")
    print(f"- Espacio de estados: {state_size}")
    print(f"- Espacio de acciones: {action_size}")
    print(f"- Pasos máximos por episodio: {max_steps}")
    print(f"- Distancia de éxito: {env.success_distance}m")
    print()

    # Crear agente optimizado
    agent = StabilizedDQNAgent(state_size, action_size)

    print("Configuración del agente:")
    print(f"- Gamma: {agent.gamma}")
    print(f"- Learning rate: {agent.learning_rate}")
    print(f"- Epsilon inicial: {agent.epsilon}")
    print(f"- Batch size: {agent.batch_size}")
    print(f"- Buffer size: {agent.memory.capacity}")
    print()

    # Entrenamiento
    num_episodes = 300
    training_results = train_stabilized_agent(env, agent, num_episodes)

    # Crear gráficos de análisis
    create_training_plots(training_results)

    # Guardar modelo
    model_filename = "optimized_dqn_aircombat.h5"
    agent.model.save(model_filename)
    print(f"\n💾 Modelo guardado como '{model_filename}'")

    # Prueba final
    test_results = run_final_test(env, agent, num_test_episodes=10)

    # Pregunta si quiere visualización 3D
    print("\n" + "=" * 70)
    show_visualization = input("¿Desea ejecutar visualización 3D de una prueba? (s/n): ").strip().lower()

    if show_visualization == 's':
        # Crear visualización simple
        print("\n🎬 Ejecutando visualización 3D...")
        visualize_test_episode(env, agent, target_position)

    print("\n🎉 Entrenamiento y evaluación completados!")
    return training_results, test_results

def visualize_test_episode(env, agent, target_pos):
    """Visualización simple de un episodio de prueba."""
    from mpl_toolkits.mplot3d import Axes3D

    # Configurar visualización
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Ejecutar episodio
    state, info = env.reset()
    trajectory = [env.uav_pos.copy()]
    done = False
    step_count = 0

    # Sin exploración
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done and step_count < env.max_steps:
        action = agent.choose_action(state, env.uav_pos, env.target_pos, training=False)
        next_state, reward, done, truncated, info = env.step(action)

        trajectory.append(env.uav_pos.copy())
        state = next_state
        step_count += 1

        if done or truncated:
            break

    # Restaurar epsilon
    agent.epsilon = original_epsilon

    # Plotear trayectoria
    trajectory = np.array(trajectory)

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            'b-', linewidth=2, alpha=0.7, label='Trayectoria UAV')

    # Posición inicial y final del UAV
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
              color='blue', s=100, marker='o', label='Inicio UAV')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
              color='cyan', s=100, marker='s', label='Final UAV')

    # Objetivo
    ax.scatter(target_pos[0], target_pos[1], target_pos[2],
              color='red', s=200, marker='^', label='Objetivo')

    # Esfera de éxito
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    r = env.success_distance
    x_sphere = target_pos[0] + r * np.outer(np.cos(u), np.sin(v))
    y_sphere = target_pos[1] + r * np.outer(np.sin(u), np.sin(v))
    z_sphere = target_pos[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='green')

    # Configuración del plot
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Trayectoria de Interceptación 3D\n'
                f'{"ÉXITO" if info["success"] else "FALLO"} - '
                f'Pasos: {step_count} - Distancia final: {info["distance"]:.2f}m')
    ax.legend()

    # Configurar límites
    all_points = np.vstack([trajectory, [target_pos]])
    margin = 20
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()