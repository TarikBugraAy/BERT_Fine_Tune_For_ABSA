#!/bin/bash

while true; do
  echo -e "\n=========================== OPTIONS ==========================="
  echo -e "Would you like to run the training phase?\n"
  echo "1) Yes, run the training phase"
  echo "2) No, skip training and go directly to testing"
  echo -e "================================================================\n"
  read -p "Enter your choice (1 or 2): " training_choice

  if [ "$training_choice" == "1" ]; then
    echo -e "\nRunning train.py...\n"
    python train.py

    if [ $? -ne 0 ]; then
      echo -e "\nTraining exited prematurely. Restarting training phase prompt.\n"
      continue
    fi
  elif [ "$training_choice" == "2" ]; then
    echo -e "\nSkipping training phase.\n"
  else
    echo -e "\nInvalid choice. Please enter 1 or 2.\n"
    continue
  fi

  while true; do
    echo -e "\n=========================== OPTIONS ==========================="
    echo -e "Choose an action:\n"
    echo "1) Run results_multi.py (for processing from input file)"
    echo "2) Run result_single.py (for inputting user's sentence for testing)"
    echo "3) Go back to the training question"
    echo "4) Exit the pipeline"
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
      echo -e "\nExiting the pipeline. Goodbye!\n"
      exit 0
    else
      echo -e "\nInvalid choice. Please enter 1, 2, 3, or 4.\n"
    fi
  done
done
