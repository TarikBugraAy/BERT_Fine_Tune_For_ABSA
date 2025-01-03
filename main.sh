#!/bin/bash

# Function to display a separator with a title
function display_section {
  local title="$1"
  echo -e "\n=========================== $title ==========================="
}

# Function to display invalid choice message
function invalid_choice {
  echo -e "\nInvalid choice. Please enter a valid option.\n"
}

# Installation Option
while true; do
  display_section "REQUIREMENTS INSTALLATION"
  echo -e "Here are the required dependencies (as per requirements.txt):\n"
  cat requirements.txt
  echo -e "\nNote: Python is assumed to be installed and is not part of the installation process.\n"
  echo "1) Yes, install packages"
  echo "2) Skip installation and go to training"
  echo "3) Exit the program"
  echo -e "====================================================================\n"
  read -p "Enter your choice (1, 2, or 3): " install_choice

  case "$install_choice" in
    1)
      echo -e "\nInstalling required packages (excluding Python)...\n"
      grep -v '^python==' requirements.txt > temp_requirements.txt
      pip install -r temp_requirements.txt
      rm temp_requirements.txt
      if [ $? -ne 0 ]; then
        echo -e "\nInstallation failed. Returning to installation options.\n"
      else
        echo -e "\nPackages installed successfully.\n"
        break
      fi
      ;;
    2)
      echo -e "\nSkipping package installation and going to training phase.\n"
      break
      ;;
    3)
      echo -e "\nExiting. Goodbye!\n"
      exit 0
      ;;
    *)
      invalid_choice
      ;;
  esac
done

# Phase Management
current_phase="training"

while true; do
  case "$current_phase" in
    training)
      display_section "TRAINING OPTIONS"
      echo "1) Yes, run the training phase"
      echo "2) Skip to testing"
      echo "3) Skip to results"
      echo "4) Exit the program"
      echo -e "================================================================\n"
      read -p "Enter your choice (1, 2, 3, or 4): " training_choice

      case "$training_choice" in
        1)
          echo -e "\nRunning train.py...\n"
          python train.py || echo -e "\nTraining exited prematurely. Returning to training phase options.\n"
          ;;
        2)
          echo -e "\nSkipping training phase and moving to testing.\n"
          current_phase="testing"
          ;;
        3)
          echo -e "\nSkipping directly to results.\n"
          current_phase="results"
          ;;
        4)
          echo -e "\nExiting. Goodbye!\n"
          exit 0
          ;;
        *)
          invalid_choice
          ;;
      esac
      ;;
    testing)
      display_section "TESTING OPTIONS"
      echo "1) Run testing using test.py"
      echo "2) Skip to results"
      echo "3) Go back to training phase"
      echo "4) Exit the program"
      echo -e "================================================================\n"
      read -p "Enter your choice (1, 2, 3, or 4): " testing_choice

      case "$testing_choice" in
        1)
          echo -e "\nRunning test.py...\n"
          python test.py
          ;;
        2)
          echo -e "\nSkipping testing phase and moving to results.\n"
          current_phase="results"
          ;;
        3)
          echo -e "\nReturning to training phase...\n"
          current_phase="training"
          ;;
        4)
          echo -e "\nExiting. Goodbye!\n"
          exit 0
          ;;
        *)
          invalid_choice
          ;;
      esac
      ;;
    results)
      display_section "RESULT OPTIONS"
      echo "1) Run results_multi.py (for processing from input file)"
      echo "2) Run result_single.py (for inputting user's sentence for testing)"
      echo "3) Go back to testing phase"
      echo "4) Go back to training phase"
      echo "5) Exit the program"
      echo -e "================================================================\n"
      read -p "Enter your choice (1, 2, 3, 4, or 5): " results_choice

      case "$results_choice" in
        1)
          echo -e "\nRunning results_multi.py...\n"
          python results_multi.py
          ;;
        2)
          echo -e "\nRunning result_single.py...\n"
          python result_single.py
          ;;
        3)
          echo -e "\nReturning to testing phase...\n"
          current_phase="testing"
          ;;
        4)
          echo -e "\nReturning to training phase...\n"
          current_phase="training"
          ;;
        5)
          echo -e "\nExiting. Goodbye!\n"
          exit 0
          ;;
        *)
          invalid_choice
          ;;
      esac
      ;;
  esac
done
