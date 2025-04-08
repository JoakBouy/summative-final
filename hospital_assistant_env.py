import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import IntEnum


class CellType(IntEnum):
    """Enum for different cell types in the environment"""
    EMPTY = 0
    AGENT = 1
    PATIENT = 2
    EMERGENCY = 3
    MEDICINE = 4
    OBSTACLE = -1
    VISITED_PATIENT = 5
    VISITED_EMERGENCY = 6
    VISITED_MEDICINE = 7


class HospitalAssistantEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=8, max_steps=200, render_mode="human"):
        super(HospitalAssistantEnv, self).__init__()
        # Environment parameters
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Action and observation spaces
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-1, high=7, shape=(self.grid_size, self.grid_size), dtype=np.int32),
            "inventory": spaces.MultiBinary(1),  # Has medicine or not
            "remaining_steps": spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.int32),
            "visited_map": spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8),
            "distance_features": spaces.Box(low=0, high=2*self.grid_size, shape=(3,), dtype=np.float32)
        })
        
        self.action_space = spaces.Discrete(5)  # 0=Up, 1=Down, 2=Left, 3=Right, 4=Interact
        
        # Initialize visualization components
        self.cell_size = 80
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None
        
        # Set up the hospital layout
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment state
        self.steps = 0
        self.has_medicine = False
        self.score = 0
        self.closest_approach_to_targets = float('inf')  # For success metrics
        self.completion_percentage = 0  # Track progress as percentage
        
        # Create an empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Initialize memory features
        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Create hospital layout first so we know valid positions
        self._create_hospital_layout()
        
        # Randomize starting position (within reasonable constraints)
        self._randomize_start_position()
        
        # Update grid with agent position
        self.grid[self.agent_pos[0], self.agent_pos[1]] = CellType.AGENT
        
        # Initialize closest approach metrics for each target type
        self.closest_patient_approach = self._get_closest_distance_to_type(CellType.PATIENT)
        self.closest_emergency_approach = self._get_closest_distance_to_type(CellType.EMERGENCY)
        self.closest_medicine_approach = self._get_closest_distance_to_type(CellType.MEDICINE)
        
        return self._get_obs(), {}

    def _randomize_start_position(self):
        # Define valid starting areas (we'll use the bottom area of the hospital)
        valid_positions = []
        
        # Bottom area, excluding obstacles and other entities
        for i in range(self.grid_size-3, self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == CellType.EMPTY:
                    valid_positions.append([i, j])
        
        # If no valid positions found, use default
        if not valid_positions:
            self.agent_pos = [self.grid_size - 2, 1]
        else:
            # Randomly select from valid positions
            self.agent_pos = valid_positions[self.np_random.choice(len(valid_positions))]

    def _create_hospital_layout(self):
        # Clear the grid
        self.grid.fill(CellType.EMPTY)
        
        # Create hallways (horizontal and vertical)
        for i in range(self.grid_size):
            # Mark center row and column as hallways (kept empty)
            if i not in [0, self.grid_size - 1]:
                for j in range(self.grid_size):
                    if abs(j - self.grid_size // 2) > 2 and abs(i - self.grid_size // 2) > 2:
                        self.grid[i, j] = CellType.OBSTACLE  # Walls between rooms
        
        # Place patient rooms
        self.patient_positions = [
            [1, 1], [1, self.grid_size - 2], 
            [self.grid_size - 2, self.grid_size - 2]
        ]
        
        # Place emergency cases (higher priority)
        self.emergency_positions = [[3, 3]]
        
        # Place medicine cabinet
        self.medicine_positions = [[self.grid_size - 2, 3]]
        
        # Additional obstacles (equipment, furniture)
        self.obstacle_positions = [
            [2, 2], [2, self.grid_size - 3], 
            [self.grid_size - 3, 2]
        ]
        
        # Set all positions on the grid
        for pos in self.patient_positions:
            self.grid[pos[0], pos[1]] = CellType.PATIENT
            
        for pos in self.emergency_positions:
            self.grid[pos[0], pos[1]] = CellType.EMERGENCY
            
        for pos in self.medicine_positions:
            self.grid[pos[0], pos[1]] = CellType.MEDICINE
            
        for pos in self.obstacle_positions:
            self.grid[pos[0], pos[1]] = CellType.OBSTACLE
        
        # Initialize visited tracking
        self.visited_patients = set()
        self.visited_emergencies = set()
        self.administered_medicine = set()

    def _get_obs(self):
        # Copy grid for observation to avoid modifying the original
        grid_obs = self.grid.copy()
        
        # Update grid with current agent position
        agent_pos_val = grid_obs[self.agent_pos[0], self.agent_pos[1]]
        grid_obs[self.agent_pos[0], self.agent_pos[1]] = CellType.AGENT
        
        # Get distance features to closest goal types
        closest_patient = self._get_closest_distance_to_type(CellType.PATIENT)
        closest_emergency = self._get_closest_distance_to_type(CellType.EMERGENCY)
        closest_medicine = self._get_closest_distance_to_type(CellType.MEDICINE)
        
        # Normalize distances
        max_distance = self.grid_size * 2
        norm_patient = closest_patient / max_distance if closest_patient < float('inf') else 1.0
        norm_emergency = closest_emergency / max_distance if closest_emergency < float('inf') else 1.0
        norm_medicine = closest_medicine / max_distance if closest_medicine < float('inf') else 1.0
        
        return {
            "grid": grid_obs,
            "inventory": np.array([int(self.has_medicine)], dtype=np.int8),
            "remaining_steps": np.array([self.max_steps - self.steps], dtype=np.int32),
            "visited_map": self.visited_cells,
            "distance_features": np.array([
                norm_patient,
                norm_emergency,
                norm_medicine
            ], dtype=np.float32)
        }

    def _get_closest_distance_to_type(self, cell_type):
        """Calculate Manhattan distance to closest cell of specified type"""
        target_positions = []
        
        if cell_type == CellType.PATIENT:
            for pos in self.patient_positions:
                if tuple(pos) not in self.visited_patients:
                    target_positions.append(pos)
        elif cell_type == CellType.EMERGENCY:
            for pos in self.emergency_positions:
                if tuple(pos) not in self.visited_emergencies:
                    target_positions.append(pos)
        elif cell_type == CellType.MEDICINE:
            if not self.has_medicine:  # Only consider medicine if we don't have it
                target_positions = self.medicine_positions
        
        if not target_positions:
            return float('inf')  # No targets of this type left
            
        distances = [abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1]) 
                    for pos in target_positions]
        return min(distances) if distances else float('inf')

    def _get_closest_goal_distance(self):
        """Calculate Manhattan distance to closest unvisited goal"""
        goals = []
        
        # Add unvisited patients
        for pos in self.patient_positions:
            if tuple(pos) not in self.visited_patients:
                goals.append(pos)
        
        # Add unvisited emergencies with higher priority
        for pos in self.emergency_positions:
            if tuple(pos) not in self.visited_emergencies:
                goals.append(pos)
        
        # Add medicine cabinet if we don't have medicine
        if not self.has_medicine:
            goals.extend(self.medicine_positions)
        
        # If we have medicine, prioritize patients we've already visited but not treated
        if self.has_medicine:
            for pos_tuple in self.visited_patients.union(self.visited_emergencies):
                if pos_tuple not in self.administered_medicine:
                    goals.append(list(pos_tuple))
        
        # Calculate closest goal
        if not goals:
            return 0  # All goals completed
            
        distances = [abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1]) 
                    for pos in goals]
        return min(distances) if distances else 0

    def _update_success_metrics(self):
        """Update metrics that track agent progress even if task isn't complete"""
        # Update closest approach metrics
        current_patient_dist = self._get_closest_distance_to_type(CellType.PATIENT)
        current_emergency_dist = self._get_closest_distance_to_type(CellType.EMERGENCY)
        current_medicine_dist = self._get_closest_distance_to_type(CellType.MEDICINE)
        
        self.closest_patient_approach = min(self.closest_patient_approach, current_patient_dist)
        self.closest_emergency_approach = min(self.closest_emergency_approach, current_emergency_dist)
        self.closest_medicine_approach = min(self.closest_medicine_approach, current_medicine_dist)
        
        # Calculate completion percentage
        total_tasks = len(self.patient_positions) + len(self.emergency_positions)
        tasks_completed = len(self.visited_patients) + len(self.visited_emergencies)
        self.completion_percentage = tasks_completed / total_tasks if total_tasks > 0 else 0

    def step(self, action):
        # Track the current step
        self.steps += 1
        reward = -0.01  # Smaller step penalty
        
        # Save previous state for shaping rewards
        old_pos = self.agent_pos.copy()
        prev_distance_to_closest_goal = self._get_closest_goal_distance()
        
        # Execute movement action
        if action < 4:  # Movement actions
            new_pos = self.agent_pos.copy()
            
            if action == 0:    # Move up
                new_pos[0] -= 1
            elif action == 1:  # Move down
                new_pos[0] += 1
            elif action == 2:  # Move left
                new_pos[1] -= 1
            elif action == 3:  # Move right
                new_pos[1] += 1
                
            # Check if the move is valid
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and 
                self.grid[new_pos[0], new_pos[1]] != CellType.OBSTACLE):
                self.agent_pos = new_pos
                
                # Update visited cells map
                self.visited_cells[self.agent_pos[0], self.agent_pos[1]] = 1
                
                # Add distance-based shaping reward
                new_distance = self._get_closest_goal_distance()
                distance_reward = prev_distance_to_closest_goal - new_distance
                reward += distance_reward * 0.5  # Scale appropriately
        
        # Handle interaction action
        elif action == 4:  # Interact
            cell_value = self.grid[self.agent_pos[0], self.agent_pos[1]]
            
            # Pick up medicine
            if cell_value == CellType.MEDICINE and not self.has_medicine:
                self.has_medicine = True
                reward += 2
            
            # Regular patient visit
            elif cell_value == CellType.PATIENT:
                pos_tuple = tuple(self.agent_pos)
                if pos_tuple not in self.visited_patients:
                    self.visited_patients.add(pos_tuple)
                    reward += 10
                    self.grid[self.agent_pos[0], self.agent_pos[1]] = CellType.VISITED_PATIENT
                    
                    # Progressive reward for visiting multiple patients
                    patient_bonus = len(self.visited_patients) * 2  # More gradual progression
                    reward += patient_bonus
                
            # Emergency patient visit
            elif cell_value == CellType.EMERGENCY:
                pos_tuple = tuple(self.agent_pos)
                if pos_tuple not in self.visited_emergencies:
                    self.visited_emergencies.add(pos_tuple)
                    reward += 15  # Slightly higher than regular patients
                    self.grid[self.agent_pos[0], self.agent_pos[1]] = CellType.VISITED_EMERGENCY
            
            # Administer medicine (to visited patients or emergencies)
            elif (cell_value in [CellType.VISITED_PATIENT, CellType.VISITED_EMERGENCY] and 
                  self.has_medicine and
                  tuple(self.agent_pos) not in self.administered_medicine):
                self.administered_medicine.add(tuple(self.agent_pos))
                self.has_medicine = False
                reward += 20  # Higher reward for completing full task
                self.grid[self.agent_pos[0], self.agent_pos[1]] = CellType.VISITED_MEDICINE
        
        # Update success metrics
        self._update_success_metrics()
        
        # Check for done condition
        done = False
        truncated = False
        
        # Win condition: All patients visited and emergencies handled
        all_patients_visited = len(self.visited_patients) == len(self.patient_positions)
        all_emergencies_handled = len(self.visited_emergencies) == len(self.emergency_positions)
        
        if all_patients_visited and all_emergencies_handled:
            done = True
            reward += 100  # Completion bonus
        
        # Time limit reached
        if self.steps >= self.max_steps:
            truncated = True
        
        # Update score
        self.score += reward
        
        # Info dictionary with additional information
        info = {
            "score": self.score,
            "visited_patients": len(self.visited_patients),
            "visited_emergencies": len(self.visited_emergencies),
            "administered_medicine": len(self.administered_medicine),
            "has_medicine": self.has_medicine,
            "all_patients_visited": all_patients_visited,
            "all_emergencies_handled": all_emergencies_handled,
            "success": all_patients_visited and all_emergencies_handled,
            "completion_percentage": self.completion_percentage,
            "closest_patient_approach": self.closest_patient_approach,
            "closest_emergency_approach": self.closest_emergency_approach,
            "closest_medicine_approach": self.closest_medicine_approach
        }
        
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        if self.render_mode is None:
            return
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Hospital Assistant Robot Environment")
            self.clock = pygame.time.Clock()
            
            # Load fonts
            self.font = pygame.font.SysFont(None, 24)
            
        if self.render_mode == "human":
            self._render_frame()
            pygame.display.flip()
            self.clock.tick(30)
            
    def _render_frame(self):
        self.window.fill((240, 240, 245))  # Light blue-gray background (hospital color)
        
        # Cell colors
        colors = {
            CellType.EMPTY: (255, 255, 255),        # White (empty)
            CellType.AGENT: (0, 120, 255),          # Blue (robot)
            CellType.PATIENT: (255, 200, 100),      # Light orange (patient)
            CellType.EMERGENCY: (255, 100, 100),    # Red (emergency)
            CellType.MEDICINE: (100, 220, 100),     # Green (medicine)
            CellType.OBSTACLE: (180, 180, 180),     # Gray (obstacle)
            CellType.VISITED_PATIENT: (200, 180, 140),  # Tan (visited patient)
            CellType.VISITED_EMERGENCY: (200, 120, 120),  # Pink (visited emergency)
            CellType.VISITED_MEDICINE: (150, 200, 150),  # Light green (medicine administered)
        }
        
        # Draw the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size, 
                    x * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                # Get cell value from the grid (not the agent position)
                cell_value = self.grid[x, y]
                
                # The agent's position gets special handling
                if [x, y] == self.agent_pos:
                    # Draw the cell first
                    pygame.draw.rect(self.window, (240, 240, 245), rect)  # Background
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)  # Border
                    
                    # Draw the agent as a circle
                    agent_color = colors[CellType.AGENT]
                    pygame.draw.circle(
                        self.window,
                        agent_color,
                        rect.center,
                        self.cell_size // 3
                    )
                    
                    # Draw medicine indicator if robot has medicine
                    if self.has_medicine:
                        pygame.draw.circle(
                            self.window,
                            (100, 220, 100),  # Green for medicine
                            (rect.center[0], rect.center[1] - self.cell_size // 5),
                            self.cell_size // 8
                        )
                else:
                    # Draw regular cell
                    cell_color = colors.get(cell_value, (200, 200, 200))
                    pygame.draw.rect(self.window, cell_color, rect)
                    pygame.draw.rect(self.window, (200, 200, 200), rect, 1)  # Cell border
                    
                    # Highlight visited cells with a small marker
                    if self.visited_cells[x, y] == 1 and cell_value == CellType.EMPTY:
                        pygame.draw.circle(
                            self.window,
                            (200, 200, 230),  # Light purple for visited
                            rect.center,
                            self.cell_size // 10
                        )
                    
                    # Add symbols or icons for different cell types
                    if cell_value == CellType.PATIENT:
                        # Patient bed symbol
                        bed_rect = pygame.Rect(
                            rect.left + rect.width // 4,
                            rect.top + rect.height // 3,
                            rect.width // 2,
                            rect.height // 3
                        )
                        pygame.draw.rect(self.window, (150, 150, 200), bed_rect)
                    elif cell_value == CellType.EMERGENCY:
                        # Red cross for emergency
                        cross_size = self.cell_size // 4
                        center_x, center_y = rect.center
                        # Horizontal line
                        pygame.draw.rect(self.window, (255, 255, 255), 
                                        (center_x - cross_size, center_y - cross_size//2, 
                                         cross_size*2, cross_size//2))
                        # Vertical line
                        pygame.draw.rect(self.window, (255, 255, 255), 
                                        (center_x - cross_size//2, center_y - cross_size, 
                                         cross_size//2, cross_size*2))
                    elif cell_value == CellType.MEDICINE:
                        # Medicine cabinet symbol
                        cabinet_rect = pygame.Rect(
                            rect.left + rect.width // 4,
                            rect.top + rect.height // 4,
                            rect.width // 2,
                            rect.height // 2
                        )
                        pygame.draw.rect(self.window, (70, 180, 70), cabinet_rect)
                        
        # Draw status information
        status_text = f"Steps: {self.steps}/{self.max_steps} | Score: {self.score:.1f}"
        text_surface = self.font.render(status_text, True, (0, 0, 0))
        self.window.blit(text_surface, (10, 10))
        
        # Draw inventory status
        med_status = "Has Medicine: Yes" if self.has_medicine else "Has Medicine: No"
        med_text = self.font.render(med_status, True, (0, 0, 0))
        self.window.blit(med_text, (10, 40))
        
        # Draw progress
        progress = f"Patients: {len(self.visited_patients)}/{len(self.patient_positions)} | "
        progress += f"Emergencies: {len(self.visited_emergencies)}/{len(self.emergency_positions)}"
        progress_text = self.font.render(progress, True, (0, 0, 0))
        self.window.blit(progress_text, (10, 70))
        
        # Draw completion percentage
        completion_text = f"Completion: {self.completion_percentage*100:.1f}%"
        completion_surface = self.font.render(completion_text, True, (0, 0, 0))
        self.window.blit(completion_surface, (10, 100))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None