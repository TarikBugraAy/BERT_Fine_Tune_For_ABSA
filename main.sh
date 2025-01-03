#!/bin/bash

# Installation Option
while true; do
  echo -e "\n=========================== INSTALLATION ==========================="
  echo -e "Here are the required dependencies (as per requirements.txt):\n"
  cat requirements.txt
  echo -e "\nNote: Python is assumed to be installed and is not part of the installation process.\n"
  echo -e "====================================================================\n"

  echo "Do you want to install the required packages?"
  echo "1) Yes, install packages"
  echo "2) No, skip installation"
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
    echo -e "\nSkipping package installation.\n"
    break
  elif [ "$install_choice" == "3" ]; then
    echo -e "\nExiting. Goodbye!\n"
    exit 0
  else
    echo -e "\nInvalid choice. Please enter 1, 2, or 3.\n"
  fi
done

# Main Menu
while true; do
  echo -e "\n=========================== OPTIONS ==========================="
  echo -e "Would you like to run the training phase?\n"
  echo "1) Yes, run the training phase"
  echo "2) No, skip training and go directly to testing"
  echo "3) Exit the program"
  echo -e "================================================================\n"
  read -p "Enter your choice (1, 2, or 3): " training_choice

  if [ "$training_choice" == "1" ]; then
    echo -e "\nRunning train.py...\n"
    python train.py

    if [ $? -ne 0 ]; then
      echo -e "\nTraining exited prematurely. Restarting training phase prompt.\n"
      continue
    fi
  elif [ "$training_choice" == "2" ]; then
    echo -e "\nSkipping training phase.\n"
  elif [ "$training_choice" == "3" ]; then
    echo -e "\nExiting. Goodbye!\n"
    exit 0
  else
    echo -e "\nInvalid choice. Please enter 1, 2, or 3.\n"
    continue
  fi

  while true; do
    echo -e "\n=========================== OPTIONS ==========================="
    echo -e "Choose an action:\n"
    echo "1) Run results_multi.py (for processing from input file)"
    echo "2) Run result_single.py (for inputting user's sentence for testing)"
    echo "3) Go back to the training question"
    echo "4) Exit the program"
    echo -e "================================================================\n"
    read -p "Enter your choice (1, 2, 3, or 4): " choice

    if [ "$choice" == "1" ]; then
      echo -e "\nRunning results_multi.py...\n"
      python results_multi.py
    elif [ "$choice" == "2" ]; then
      echo -e "\nRunning result_single.py...\n"
      python result_single.py
    elif [ "$choice" == "3" ]; then
      echo -e "\nReturning to the training phase question...\n"
      break
    elif [ "$choice" == "4" ]; then
      echo -e "\nExiting. Goodbye!\n"
      exit 0
    else
      echo -e "\nInvalid choice. Please enter 1, 2, 3, or 4.\n"
    fi
  done
done
