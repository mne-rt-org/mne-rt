'''
#!/bin/bash
set -e

echo "=== Starting Antares Pipeline ==="
echo ""
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ant-test

## stage 1
USER_DATA=$(python antares_1.py)
IFS=',' read -r SUBJECT_ID AGE SEX VISIT <<< "$USER_DATA"

## stage 2
python antares_2.py --subject_id "$SUBJECT_ID" --visit "$VISIT"

## stage 3
python antares_3.py --subject_id "$SUBJECT_ID" --age "$AGE" --sex "$SEX" --visit "$VISIT"

## stage 4
python antares_4.py --subject_id "$SUBJECT_ID" --visit "$VISIT"

## stage 5
python antares_5.py --subject_id "$SUBJECT_ID" --visit "$VISIT"

echo ""
echo "=== Done ==="

'''
#!/bin/bash
set -e
echo "=== Starting Antares Pipeline ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ant-test

# Stage 1: Get user data first
USER_DATA=$(python antares_1.py)
IFS=',' read -r SUBJECT_ID AGE SEX VISIT <<< "$USER_DATA"

echo "User data collected: $SUBJECT_ID, $AGE, $SEX, $VISIT"

# Stage 2: Launch GUI with the user data as arguments
python antares_gui.py "$SUBJECT_ID" "$AGE" "$SEX" "$VISIT"

echo "=== Done ==="
