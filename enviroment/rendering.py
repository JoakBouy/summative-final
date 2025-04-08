import gymnasium as gym
import pygame
import time
import sys
import numpy as np
from gymnasium import spaces

# Import the environment class
from hospital_assistant_env import HospitalAssistantEnv, CellType

def main():
    """
    Main function to create and render the Hospital Assistant Environment
    """
    # Initialize pygame first
    pygame.init()
    
    # Create the environment
    env = HospitalAssistantEnv(grid_size=8, max_steps=200, render_mode="human")
    
    # Reset the environment to initialize
    observation, info = env.reset()
    
    # Print instructions
    print("Hospital Assistant Environment")
    print("Controls:")
    print("  Arrow Keys: Move (Up, Down, Left, Right)")
    print("  Space: Interact")
    print("  Q: Quit")
    print("  R: Reset Environment")
    
    # Display initial metrics
    print("\nInitial Environment State:")
    print(f"Completion Percentage: {info.get('completion_percentage', 0)*100:.1f}%")
    print(f"Closest approaches - Patient: {observation['distance_features'][0]:.2f}, " 
          f"Emergency: {observation['distance_features'][1]:.2f}, "
          f"Medicine: {observation['distance_features'][2]:.2f}")
    
    # Main game loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle keyboard input
            elif event.type == pygame.KEYDOWN:
                action = None
                
                # Movement
                if event.key == pygame.K_UP:
                    action = 0  # Move up
                elif event.key == pygame.K_DOWN:
                    action = 1  # Move down
                elif event.key == pygame.K_LEFT:
                    action = 2  # Move left
                elif event.key == pygame.K_RIGHT:
                    action = 3  # Move right
                elif event.key == pygame.K_SPACE:
                    action = 4  # Interact
                elif event.key == pygame.K_q:
                    running = False  # Quit
                elif event.key == pygame.K_r:
                    observation, info = env.reset()  # Reset
                    print("\nEnvironment Reset!")
                    print(f"Completion Percentage: {info.get('completion_percentage', 0)*100:.1f}%")
                
                # If a valid action was chosen, step the environment
                if action is not None:
                    observation, reward, done, truncated, info = env.step(action)
                    
                    # Print information about the step
                    print(f"\nAction: {action}, Reward: {reward:.2f}")
                    print(f"Step: {env.steps}/{env.max_steps}, Score: {info['score']:.2f}")
                    print(f"Completion: {info['completion_percentage']*100:.1f}%, " 
                          f"Patients: {info['visited_patients']}/{len(env.patient_positions)}, "
                          f"Emergencies: {info['visited_emergencies']}/{len(env.emergency_positions)}")
                    print(f"Medicine status: {'Carrying' if info['has_medicine'] else 'Not carrying'}, "
                          f"Administered: {info['administered_medicine']}")
                    
                    # Check if episode is done
                    if done or truncated:
                        print("\n--- Episode finished! ---")
                        if info.get("success", False):
                            print("Mission accomplished! All patients and emergencies handled.")
                        elif truncated:
                            print("Time limit reached.")
                        print(f"Final score: {info['score']:.2f}")
        
        # Render the environment
        env.render()
        
        # Cap the frame rate
        clock.tick(30)
    
    # Clean up
    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()