#!/bin/bash

while true; do
  echo "======================================================"
  echo "Would you like to run the training phase?"
  echo "1) Yes, run the training phase"
  echo "2) No, skip training and go directly to inference"
  read -p "Enter your choice (1 or 2): " training_choice
  echo "======================================================"

  if [ "$training_choice" == "1" ]; then
    echo "Running train.py..."
    python train.py

    if [ $? -ne 0 ]; then
      echo "Training exited prematurely. Restarting training phase prompt."
      continue
    fi
  elif [ "$training_choice" == "2" ]; then
    echo "Skipping training phase."
  else
    echo "Invalid choice. Please enter 1 or 2."
    continue
  fi


  while true; do
    echo "======================================================"
    echo "Choose an action:"
    echo "1) Run results_multi.py (batch processing from CSV)"
    echo "2) Run result_single.py (single-sentence testing)"
    echo "3) Go back to the training question"
    echo "4) Exit the pipeline"
    read -p "Enter your choice (1, 2, 3, or 4): " choice
    echo "======================================================"

    if [ "$choice" == "1" ]; then
      echo "Running results_multi.py..."
      python results_multi.py
    elif [ "$choice" == "2" ]; then
      echo "Running result_single.py..."
      python result_single.py
    elif [ "$choice" == "3" ]; then
      echo "Returning to the training phase question..."
      break
    elif [ "$choice" == "4" ]; then
      echo "Exiting the pipeline. Goodbye!"
      exit 0
    else
      echo "Invalid choice. Please enter 1, 2, 3, or 4."
    fi
  done
done
