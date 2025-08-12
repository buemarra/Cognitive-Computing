# Prueba de Concepto: Inferencia Activa para Sistemas de Defensa
# Simulación de UAV usando pymdp para combate aéreo
# ! pip install inferactively-pymdp

# Prueba de Concepto: Inferencia Activa para Sistemas de Defensa
# Simulación de UAV usando pymdp para combate aéreo

import numpy as np
import matplotlib.pyplot as plt
from pymdp.agent import Agent
from pymdp import utils

# ===== CONFIGURACIÓN DEL ENTORNO =====

def setup_environment():
    """
    Configura el entorno de combate aéreo con estados, observaciones y acciones
    """
    # Estados del mundo
    num_states = [3, 3]  # [pos_x, pos_y] cada uno con 3 valores posibles
    
    # Observaciones posibles
    num_obs = [3, 3]  # [rel_x, rel_y] cada uno con 3 valores posibles
    
    # Acciones posibles
    num_controls = [3, 3]  # [mov_x, mov_y] cada uno con 3 valores posibles
    
    return num_states, num_obs, num_controls

# ===== MATRICES DEL MODELO GENERATIVO =====

def create_observation_model(num_obs, num_states):
    """
    Crea la matriz A (modelo de observación)
    A[m][o, s] = P(o_m | s) donde m es la modalidad
    """
    A = utils.obj_array_zeros([[num_obs[0], num_states[0]], 
                               [num_obs[1], num_states[1]]])
    
    # Modalidad 0: observación rel_x dado pos_x
    # Si estoy en la izquierda (estado 0), es más probable observar "izq" (obs 0)
    A[0] = np.array([
        [0.8, 0.1, 0.05],  # P(obs=izq | pos_x=izq, centro, der)
        [0.15, 0.8, 0.15], # P(obs=centro | pos_x=izq, centro, der)
        [0.05, 0.1, 0.8]   # P(obs=der | pos_x=izq, centro, der)
    ])
    
    # Modalidad 1: observación rel_y dado pos_y
    A[1] = np.array([
        [0.8, 0.1, 0.05],  # P(obs=abajo | pos_y=abajo, medio, arriba)
        [0.15, 0.8, 0.15], # P(obs=medio | pos_y=abajo, medio, arriba)
        [0.05, 0.1, 0.8]   # P(obs=arriba | pos_y=abajo, medio, arriba)
    ])
    
    # Normalizar para asegurar que suman a 1
    for i in range(len(A)):
        A[i] = A[i] / A[i].sum(axis=0, keepdims=True)
    
    return A

def create_transition_model(num_states, num_controls):
    """
    Crea la matriz B (modelo de transición)
    B[f][s', s, a] = P(s'_f | s_f, a_f) donde f es el factor de estado
    """
    B = utils.obj_array_zeros([[num_states[0], num_states[0], num_controls[0]], 
                               [num_states[1], num_states[1], num_controls[1]]])
    
    # Factor 0: transiciones en pos_x basadas en mov_x
    # Acción 0: mover izquierda
    B[0][:, :, 0] = np.array([
        [0.8, 0.6, 0.2],  # P(pos_x'=izq | pos_x=izq,centro,der, mov_x=izq)
        [0.15, 0.3, 0.4], # P(pos_x'=centro | pos_x=izq,centro,der, mov_x=izq)
        [0.05, 0.1, 0.4]  # P(pos_x'=der | pos_x=izq,centro,der, mov_x=izq)
    ])
    
    # Acción 1: mantener centro
    B[0][:, :, 1] = np.array([
        [0.6, 0.2, 0.1],  # P(pos_x'=izq | pos_x=izq,centro,der, mov_x=centro)
        [0.3, 0.6, 0.3],  # P(pos_x'=centro | pos_x=izq,centro,der, mov_x=centro)
        [0.1, 0.2, 0.6]   # P(pos_x'=der | pos_x=izq,centro,der, mov_x=centro)
    ])
    
    # Acción 2: mover derecha
    B[0][:, :, 2] = np.array([
        [0.4, 0.1, 0.05], # P(pos_x'=izq | pos_x=izq,centro,der, mov_x=der)
        [0.4, 0.3, 0.15], # P(pos_x'=centro | pos_x=izq,centro,der, mov_x=der)
        [0.2, 0.6, 0.8]   # P(pos_x'=der | pos_x=izq,centro,der, mov_x=der)
    ])
    
    # Factor 1: transiciones en pos_y basadas en mov_y
    # Acción 0: mover abajo
    B[1][:, :, 0] = np.array([
        [0.8, 0.6, 0.2],  # P(pos_y'=abajo | pos_y=abajo,medio,arriba, mov_y=abajo)
        [0.15, 0.3, 0.4], # P(pos_y'=medio | pos_y=abajo,medio,arriba, mov_y=abajo)
        [0.05, 0.1, 0.4]  # P(pos_y'=arriba | pos_y=abajo,medio,arriba, mov_y=abajo)
    ])
    
    # Acción 1: mantener medio
    B[1][:, :, 1] = np.array([
        [0.6, 0.2, 0.1],  # P(pos_y'=abajo | pos_y=abajo,medio,arriba, mov_y=medio)
        [0.3, 0.6, 0.3],  # P(pos_y'=medio | pos_y=abajo,medio,arriba, mov_y=medio)
        [0.1, 0.2, 0.6]   # P(pos_y'=arriba | pos_y=abajo,medio,arriba, mov_y=medio)
    ])
    
    # Acción 2: mover arriba
    B[1][:, :, 2] = np.array([
        [0.4, 0.1, 0.05], # P(pos_y'=abajo | pos_y=abajo,medio,arriba, mov_y=arriba)
        [0.4, 0.3, 0.15], # P(pos_y'=medio | pos_y=abajo,medio,arriba, mov_y=arriba)
        [0.2, 0.6, 0.8]   # P(pos_y'=arriba | pos_y=abajo,medio,arriba, mov_y=arriba)
    ])
    
    # Normalizar para asegurar que suman a 1
    for i in range(len(B)):
        for j in range(B[i].shape[2]):
            B[i][:, :, j] = B[i][:, :, j] / B[i][:, :, j].sum(axis=0, keepdims=True)
    
    return B

def create_preference_model(num_obs):
    """
    Crea la matriz C (preferencias/objetivos)
    C[m][o] = preferencia para observación o en modalidad m
    """
    C = utils.obj_array_zeros([num_obs[0], num_obs[1]])
    
    # Modalidad 0: preferencia por mantener el objetivo en el centro (rel_x)
    C[0] = np.array([-2.0, 2.0, -2.0])  # [izq, centro, der]
    
    # Modalidad 1: preferencia por mantener el objetivo en el centro (rel_y)
    C[1] = np.array([-2.0, 2.0, -2.0])  # [abajo, medio, arriba]
    
    return C

def create_prior_beliefs(num_states):
    """
    Crea las creencias iniciales D
    """
    D = utils.obj_array_uniform(num_states)
    return D

# ===== SIMULACIÓN DEL COMBATE =====

def simulate_target_movement(step, target_state):
    """
    Simula el movimiento del objetivo enemigo
    """
    # El objetivo se mueve de forma semi-aleatoria
    if step % 3 == 0:  # Cambio de dirección cada 3 pasos
        target_state[0] = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])  # pos_x
        target_state[1] = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])  # pos_y
    
    return target_state

def generate_observations(uav_state, target_state, noise_level=0.1):
    """
    Genera observaciones basadas en las posiciones relativas
    """
    # Posición relativa del objetivo respecto al UAV
    rel_x = target_state[0] - uav_state[0]  # -1, 0, 1
    rel_y = target_state[1] - uav_state[1]  # -1, 0, 1
    
    # Mapear a observaciones discretas con ruido
    obs_x = np.clip(rel_x + 1, 0, 2)  # Mapear [-1,0,1] a [0,1,2]
    obs_y = np.clip(rel_y + 1, 0, 2)
    
    # Añadir ruido
    if np.random.random() < noise_level:
        obs_x = np.random.choice([0, 1, 2])
    if np.random.random() < noise_level:
        obs_y = np.random.choice([0, 1, 2])
    
    return [obs_x, obs_y]

def run_combat_simulation(steps=20):
    """
    Ejecuta la simulación completa de combate aéreo
    """
    print("Iniciando simulación de combate aéreo con Inferencia Activa...")
    
    # Configurar entorno
    num_states, num_obs, num_controls = setup_environment()
    
    # Crear matrices del modelo
    A = create_observation_model(num_obs, num_states)
    B = create_transition_model(num_states, num_controls)
    C = create_preference_model(num_obs)
    D = create_prior_beliefs(num_states)
    
    # Verificar dimensiones
    print(f"Dimensiones del modelo:")
    print(f"  Estados: {num_states}")
    print(f"  Observaciones: {num_obs}")
    print(f"  Controles: {num_controls}")
    print(f"  Matriz A: {[A[i].shape for i in range(len(A))]}")
    print(f"  Matriz B: {[B[i].shape for i in range(len(B))]}")
    
    # Crear agente con parámetros básicos
    try:
        agent = Agent(
            A=A, 
            B=B, 
            C=C, 
            D=D,
            policy_len=3  # Planificación a 3 pasos
        )
        print("Agente creado exitosamente!")
    except Exception as e:
        print(f"Error creando agente: {e}")
        # Intentar con parámetros mínimos
        agent = Agent(A=A, B=B, C=C, D=D)
        print("Agente creado con parámetros mínimos!")
    
    # Estados iniciales
    uav_state = [1, 1]  # Empezar en el centro
    target_state = [1, 1]  # Objetivo también en el centro
    
    # Historiales para visualización
    history = {
        'uav_states': [],
        'target_states': [],
        'observations': [],
        'actions': [],
        'beliefs': []
    }
    
    # Simulación paso a paso
    for step in range(steps):
        print(f"\\nPaso {step + 1}/{steps}")
        
        # Simular movimiento del objetivo
        target_state = simulate_target_movement(step, target_state.copy())
        
        # Generar observaciones
        obs = generate_observations(uav_state, target_state)
        
        # Asegurar que las observaciones son del tipo correcto
        if isinstance(obs, list):
            obs = [np.array([o], dtype=int) for o in obs]
        
        try:
            # El agente infiere estados y selecciona acción
            qs = agent.infer_states(obs)
            action = agent.sample_action()
            
            # Convertir acción a lista si es necesario
            if hasattr(action, '__iter__') and not isinstance(action, (str, bytes)):
                action = list(action)
            else:
                action = [action, action]  # Duplicar si es un solo valor
                
        except Exception as e:
            print(f"Error en inferencia/acción: {e}")
            # Acción aleatoria como fallback
            action = [np.random.choice([0, 1, 2]), np.random.choice([0, 1, 2])]
            qs = [np.ones(3)/3, np.ones(3)/3]  # Creencias uniformes
        
        # Actualizar estado del UAV basado en la acción
        # Acción es una lista [mov_x, mov_y]
        if action[0] == 0:  # mover izquierda
            uav_state[0] = max(0, uav_state[0] - 1)
        elif action[0] == 2:  # mover derecha
            uav_state[0] = min(2, uav_state[0] + 1)
        # action[0] == 1 significa mantener posición x
        
        if action[1] == 0:  # mover abajo
            uav_state[1] = max(0, uav_state[1] - 1)
        elif action[1] == 2:  # mover arriba
            uav_state[1] = min(2, uav_state[1] + 1)
        # action[1] == 1 significa mantener posición y
        
        # Guardar en historial
        history['uav_states'].append(uav_state.copy())
        history['target_states'].append(target_state.copy())
        history['observations'].append([obs[0][0] if hasattr(obs[0], '__iter__') else obs[0], 
                                       obs[1][0] if hasattr(obs[1], '__iter__') else obs[1]])
        history['actions'].append(action.copy())
        
        # Asegurar que qs tenga el formato correcto para el historial
        if isinstance(qs, list):
            history['beliefs'].append([np.array(qs[0]), np.array(qs[1])])
        else:
            history['beliefs'].append([np.ones(3)/3, np.ones(3)/3])
        
        # Mostrar información del paso
        pos_names = ['izq', 'centro', 'der']
        obs_display = history['observations'][-1]
        print(f"  UAV pos: [{pos_names[uav_state[0]]}, {pos_names[uav_state[1]]}]")
        print(f"  Target pos: [{pos_names[target_state[0]]}, {pos_names[target_state[1]]}]")
        print(f"  Observación: [{pos_names[obs_display[0]]}, {pos_names[obs_display[1]]}]")
        print(f"  Acción: [{pos_names[action[0]]}, {pos_names[action[1]]}]")
        
        # Calcular distancia al objetivo
        distance = abs(uav_state[0] - target_state[0]) + abs(uav_state[1] - target_state[1])
        print(f"  Distancia al objetivo: {distance}")
    
    return history

def visualize_combat(history):
    """
    Visualiza los resultados de la simulación
    """
    steps = len(history['uav_states'])
    
    # Extraer coordenadas
    uav_x = [state[0] for state in history['uav_states']]
    uav_y = [state[1] for state in history['uav_states']]
    target_x = [state[0] for state in history['target_states']]
    target_y = [state[1] for state in history['target_states']]
    
    # Crear visualización
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Trayectorias
    plt.subplot(2, 2, 1)
    plt.plot(uav_x, uav_y, 'bo-', label='UAV (Agente)', linewidth=2, markersize=6)
    plt.plot(target_x, target_y, 'rx--', label='Objetivo', linewidth=2, markersize=6)
    plt.xlabel('Posición X')
    plt.ylabel('Posición Y')
    plt.title('Trayectorias en Combate Aéreo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([0, 1, 2], ['Izq', 'Centro', 'Der'])
    plt.yticks([0, 1, 2], ['Abajo', 'Medio', 'Arriba'])
    
    # Subplot 2: Distancia al objetivo
    plt.subplot(2, 2, 2)
    distances = [abs(uav_x[i] - target_x[i]) + abs(uav_y[i] - target_y[i]) 
                for i in range(steps)]
    plt.plot(distances, 'g-', linewidth=2)
    plt.xlabel('Paso')
    plt.ylabel('Distancia Manhattan')
    plt.title('Distancia UAV-Objetivo')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Acciones tomadas
    plt.subplot(2, 2, 3)
    actions_x = [action[0] for action in history['actions']]
    actions_y = [action[1] for action in history['actions']]
    plt.scatter(range(steps), actions_x, c='blue', alpha=0.7, label='Mov X')
    plt.scatter(range(steps), actions_y, c='red', alpha=0.7, label='Mov Y')
    plt.xlabel('Paso')
    plt.ylabel('Acción')
    plt.title('Acciones del UAV')
    plt.legend()
    plt.yticks([0, 1, 2], ['Izq/Abajo', 'Centro/Medio', 'Der/Arriba'])
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Creencias del agente (certeza promedio)
    plt.subplot(2, 2, 4)
    belief_certainty = []
    for beliefs in history['beliefs']:
        # Calcular la certeza como el máximo de cada creencia
        certainty_x = np.max(beliefs[0])
        certainty_y = np.max(beliefs[1])
        belief_certainty.append((certainty_x + certainty_y) / 2)
    
    plt.plot(belief_certainty, 'm-', linewidth=2)
    plt.xlabel('Paso')
    plt.ylabel('Certeza Promedio')
    plt.title('Confianza en las Creencias')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas finales
    print(f"\\n=== ESTADÍSTICAS FINALES ===")
    print(f"Pasos totales: {steps}")
    print(f"Distancia final: {distances[-1]}")
    print(f"Distancia promedio: {np.mean(distances):.2f}")
    print(f"Certeza promedio: {np.mean(belief_certainty):.3f}")
    
    # Conteo de acciones
    action_counts = {}
    for action in history['actions']:
        action_tuple = tuple(action)
        action_counts[action_tuple] = action_counts.get(action_tuple, 0) + 1
    
    print(f"\\nAcciones más frecuentes:")
    action_names = {0: 'izq/abajo', 1: 'centro/medio', 2: 'der/arriba'}
    for action_tuple, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        action_str = f"[{action_names[action_tuple[0]]}, {action_names[action_tuple[1]]}]"
        print(f"  {action_str}: {count} veces ({count/steps*100:.1f}%)")

# ===== FUNCIÓN PRINCIPAL =====

def main():
    """
    Función principal que ejecuta la simulación completa
    """
    print("=== PRUEBA DE CONCEPTO: INFERENCIA ACTIVA PARA DEFENSA ===\\n")
    
    try:
        # Ejecutar simulación
        history = run_combat_simulation(steps=15)
        
        # Visualizar resultados
        print("\\nGenerando visualizaciones...")
        visualize_combat(history)
        
        print("\\n¡Simulación completada exitosamente!")
        
    except Exception as e:
        print(f"Error durante la simulación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()