#!/bin/bash

# Installation Option
while true; do
  echo -e "\n=========================== REQUIREMENTS INSTALLATION ==========================="
  echo -e "Here are the required dependencies (as per requirements.txt):\n"
  cat requirements.txt
  echo -e "\nNote: Python is assumed to be installed and is not part of the installation process.\n"
  echo -e "====================================================================\n"

  echo "Do you want to install the required packages?"
  echo "1) Yes, install packages"
  echo "2) Skip installation and go to training"
  echo "3) Exit the program"
  echo -e "====================================================================\n"
  read -p "Enter your choice (1, 2, or 3): " install_choice

  if [ "$install_choice" == "1" ]; then
    echo -e "\nInstalling required packages (excluding Python)...\n"
    grep -v '^python==' requirements.txt > temp_requirements.txt
    pip install -r temp_requirements.txt
    rm temp_requirements.txt

    if [ $? -ne 0 ]; then
      echo -e "\nInstallation failed. Returning to installation options.\n"
      continue
    fi
    echo -e "\nPackages installed successfully.\n"
    break
  elif [ "$install_choice" == "2" ]; then
    echo -e "\nSkipping package installation and going to training phase.\n"
    break
  elif [ "$install_choice" == "3" ]; then
    echo -e "\nExiting. Goodbye!\n"
    exit 0
  else
    echo -e "\nInvalid choice. Please enter 1, 2, or 3.\n"
  fi
done

# Phase Management
current_phase="training"

while true; do
  if [ "$current_phase" == "training" ]; then
    # Training Options
    echo -e "\n=========================== TRAINING OPTIONS ==========================="
    echo -e "Would you like to run the training phase?\n"
    echo "1) Yes, run the training phase"
    echo "2) Skip to testing"
    echo "3) Skip to results"
    echo "4) Exit the program"
    echo -e "================================================================\n"
    read -p "Enter your choice (1, 2, 3, or 4): " training_choice

    if [ "$training_choice" == "1" ]; then
      echo -e "\nRunning train.py...\n"
      python train.py

      if [ $? -ne 0 ]; then
        echo -e "\nTraining exited prematurely. Returning to training phase options.\n"
        continue
      fi
    elif [ "$training_choice" == "2" ]; then
      echo -e "\nSkipping training phase and moving to testing.\n"
      current_phase="testing"
      continue
    elif [ "$training_choice" == "3" ]; then
      echo -e "\nSkipping directly to results.\n"
      current_phase="results"
      continue
    elif [ "$training_choice" == "4" ]; then
      echo -e "\nExiting. Goodbye!\n"
      exit 0
    else
      echo -e "\nInvalid choice. Please enter 1, 2, 3, or 4.\n"
      continue
    fi
    current_phase="testing"
  fi

  if [ "$current_phase" == "testing" ]; then
    # Testing Options
    echo -e "\n=========================== TESTING OPTIONS ==========================="
    echo -e "Choose an action:\n"
    echo "1) Run testing using test.py"
    echo "2) Skip to results"
    echo "3) Go back to training phase"
    echo "4) Exit the program"
    echo -e "================================================================\n"
    read -p "Enter your choice (1, 2, 3, or 4): " testing_choice

    if [ "$testing_choice" == "1" ]; then
      echo -e "\nRunning test.py...\n"
      python test.py
    elif [ "$testing_choice" == "2" ]; then
      echo -e "\nSkipping testing phase and moving to results.\n"
      current_phase="results"
      continue
    elif [ "$testing_choice" == "3" ]; then
      echo -e "\nReturning to training phase...\n"
      current_phase="training"
      continue
    elif [ "$testing_choice" == "4" ]; then
      echo -e "\nExiting. Goodbye!\n"
      exit 0
    else
      echo -e "\nInvalid choice. Please enter 1, 2, 3, or 4.\n"
      continue
    fi
    current_phase="results"
  fi

  if [ "$current_phase" == "results" ]; then
    # Results Options
    echo -e "\n=========================== RESULT OPTIONS ==========================="
    echo -e "Choose an action:\n"
    echo "1) Run results_multi.py (for processing from input file)"
    echo "2) Run result_single.py (for inputting user's sentence for testing)"
    echo "3) Go back to testing phase"
    echo "4) Go back to training phase"
    echo "5) Exit the program"
    echo -e "================================================================\n"
    read -p "Enter your choice (1, 2, 3, 4, or 5): " choice

    if [ "$choice" == "1" ]; then
      echo -e "\nRunning results_multi.py...\n"
      python results_multi.py
    elif [ "$choice" == "2" ]; then
      echo -e "\nRunning result_single.py...\n"
      python result_single.py
    elif [ "$choice" == "3" ]; then
      echo -e "\nReturning to testing phase...\n"
      current_phase="testing"
      continue
    elif [ "$choice" == "4" ]; then
      echo -e "\nReturning to training phase...\n"
      current_phase="training"
      continue
    elif [ "$choice" == "5" ]; then
      echo -e "\nExiting. Goodbye!\n"
      exit 0
    else
      echo -e "\nInvalid choice. Please enter 1, 2, 3, 4, or 5.\n"
    fi
  fi
done
