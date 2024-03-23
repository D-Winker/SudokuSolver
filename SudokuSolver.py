# Sudoku Solver
#
# Reads the file "sudoku.txt" (from [dimitri] https://github.com/dimitri/sudoku/tree/master)
# and solves them.
#
# Daniel Winker, March 22, 2024
# TODO: So far rules 1, 2, 3, 4, 5, 6, 7, 8 have been seen to work. That's all of them!
# TODO: Make rule 6 highlight all relevant numbers when pretty plotting. Done!
# TODO: Save off plots as it solves and make an animation.
# TODO: Solve through optimization.
# TODO: Try solving with linear back projection. I project back sum of the unused row, column, and box values. Normalize. Scale 1 - 9.
# TODO: Could Loopy BP or something similar be used here?
# TODO: Use dad's Monte Carlo idea. Maybe that plus wavefunction collapse?
# TODO: Try the recursive guesser as a solver by itself, no other rules. (...or not. The recursive part is written, but it would require a different checker. Hmm...make all rows, columns, and boxes into sets, and check the size of each?)
#
# Solver 1 is approximately how I would solve a sudoku by hand. 
# It solved all 50 Sudoku in 834 seconds, or 16.7 seconds per Sudoku.

import time
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from itertools import combinations
from functools import reduce
from matplotlib.patches import Rectangle
from copy import deepcopy

# Read all of the text from the file, one line at a time
textFile = open("sudoku.txt", 'r')
fileContent = textFile.readlines()

# Populate 9x9 numpy arrays with the sudoku
sudoku = []  # A list to hold each 9x9 sudoku array
sudokuArr = np.zeros((9,9))  # Initialize a sudoku array
rowCount = 0  # Track which row of the grid the line of text corresponds to
for line in fileContent:
    if "Grid" in line:
        if not np.all(sudokuArr == np.zeros((9,9))):
            sudoku.append(sudokuArr)  # Add the previous sudoku to our list
            sudokuArr = np.zeros((9,9))  # Start a new sudoku
            rowCount = 0
    
    else:
        for col in range(9):
            sudokuArr[rowCount, col] = int(line[col])
        rowCount += 1

sudoku.append(sudokuArr)  # Add the previous sudoku to our list


def prettyPlot(_sudoku, title="", prevSudoku=None, currRows=[], currCols=[], currValIndexes=[], rule=-1, stepCount=-1):
    """
    This creates a visual to understand how the solver is working.
    Written with ChatGPT.
    """
    _sudoku = _sudoku.astype(int)

    if prevSudoku is None:
        prevSudoku = _sudoku
    else:
        prevSudoku = prevSudoku.astype(int)
        if np.all(prevSudoku == _sudoku):
            return 1

    # DEBUG, skip the rules that have already been tested
    # We have to skip after the above check, which rejects checks that don't result in a change
    if rule in [1, 2, 3, 4, 5, 6, 7, 8]:
        return 0

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Iterate through each cell in the 9x9 grid
    for row in range(9):
        for col in range(9):
            # Extract the 3x3 subgrid for the current cell (i.e. the potential values for the square)
            subgrid = _sudoku[row, col]
            prevSubgrid = prevSudoku[row, col]

            # Iterate through each cell in the 3x3 subgrid
            for subrow in range(3):
                for subcol in range(3):
                    valueIndex = subrow * 3 + subcol
                    value = subgrid[valueIndex]
                    prevValue = prevSubgrid[valueIndex]  

                    xcoord = col * 3 + subcol + 0.5
                    ycoord = 27 - (row * 3 + subrow + 0.6)
                    numberSize = 12
                    if np.count_nonzero(subgrid) == 1 and np.count_nonzero(prevSubgrid) == 1:
                        xcoord = col * 3 + 1.4
                        ycoord = 27 - (row * 3 + 1.6)
                        numberSize = 30

                    # Draw the value in the 3x3 subgrid cell
                    if ((row in currRows and col in currCols and valueIndex in currValIndexes and rule not in (3, 4, 7)) or 
                        (rule in (3, 4, 7) and (row, col) in zip(currRows, currCols))) and (value == prevValue and prevValue != 0):
                        # Draw the current value of interest in green, unless the current value was removed
                        ax.text(xcoord, ycoord, str(value), fontsize=numberSize+4, ha='center', va='center', color='green', weight='bold') 

                    elif value == prevValue:
                        # If the value didn't change...
                        if value == 0:
                            # and it's zero, don't show it
                            pass  # ax.text(xcoord, ycoord, str(value), fontsize=numberSize, ha='center', va='center', color='gray')

                        else:
                            # if it's nonzero, show it in 
                            ax.text(xcoord, ycoord, str(value), fontsize=numberSize, ha='center', va='center', color='black')
                    
                    else:
                        # If the value did change (i.e. to zero, it was removed), show it in red
                        ax.text(xcoord, ycoord, str(prevValue), fontsize=numberSize+4, ha='center', va='center', color='blue', weight='bold')
                    
            # Draw a rectangle around the 3x3 subgrid
            rect = Rectangle((col * 3, 27 - row * 3), 3, -3, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            if row % 3 == 0 and col % 3 == 0:
                # Draw thicker rectangles around the 3x3 subgrid of subgrids
                rect = Rectangle((col * 3, 27 - row * 3), 9, -9, linewidth=3, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

    # Set axis limits
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(title)

    # Show the plot
    plt.show()

    return 0



# Sudoku basics
# Each 3x3 grid can only have one of each number
# Each row can only have one of each number
# Each column can only have one of each number
# Each cell can hold any of the numbers

# Simple sudoku can be solved by 'looking at where a number can go' - 
# you consider which numbers aren't already present in a 3x3 box, then
# consider where those numbers can be placed. To solve more difficult 
# sudoku I also consider what numbers a cell can hold*. If a cell can
# only hold one value, then even if other cells could also hold that 
# value (given the current knowledge of the board), then that one cell
# must hold that value.
# *I assume this is a normal approach, but I want to write this without looking up techniques.

def solver1(_sudoku, stepCounter=0, recursionDepth=0, itr=0):
    """
    This is more or less how I solve Sudoku by hand. I didn't look up any Sudoku solving techniques to write
    this function, but I did run across a Numberphile video that taught me a new rule (below as Rule 7), which
    I included here. It's somewhat niche, but it does come up.
    This function uses 7 rules and if they don't make any progress, guesses. The basic concept is: every square
    can contain the values 1 - 9 until proven otherwise. One by one, check the 7 constraints to see what values
    should be removed, or in some cases, what value must be in a square.
    """

    # Determine whether this is the initial call, or a recursive call (which will have more information)
    if len(_sudoku.shape) == 2:
        #print("Initial setup")
        # Instantiate the array of all possible values (1-9 for each square)
        # I'm sure there's a pythonic way to do this, but in the spirit of the naive method
        potentialValues = np.zeros((9,9,9))
        for row in range(9):
            for col in range(9):
                for val in range(1,10):
                    potentialValues[row, col, val-1] = val
        
        # Handle the initial values
        for row in range(9):
            for col in range(9):
                val = _sudoku[row,col]
                if val != 0:
                    potentialValues[row, col, potentialValues[row,col]!=val] = 0  # Set everything except 'val' to zero
        
        #prettyPlot(potentialValues, f"Step: {stepCounter}, initial state")
    
    elif len(_sudoku.shape) == 3:
        #print(f"Recursed down, depth: {recursionDepth}")
        potentialValues = _sudoku


    # Iterate over every square until each square only contains one nonzero value
    solved = False
    while not solved:
        prevStepCounter = stepCounter  # To track if anything changes
        for row in range(9):
            for col in range(9):
                for valueIndex in range(9):
                    
                    val = potentialValues[row, col, valueIndex]
                    if val == 0:
                        continue

                    # Get all of the current potential values from this box
                    startRow = 3 * (row // 3)
                    startCol = 3 * (col // 3)
                    currBox = potentialValues[startRow:startRow+3,startCol:startCol+3,:]
                    currSquare = potentialValues[row, col, :]

                    # Rule 1
                    # Check the values in the box
                    # If the current value in this square doesn't appear anywhere else
                    # in this box, then all other values in this square should be assigned 0
                    if (val not in currBox[np.arange(3)!=row%3, :, :]) and (val not in currBox[:, np.arange(3)!=col%3, :]):
                        prevValues = deepcopy(potentialValues)
                        potentialValues[row, col, np.arange(9)!=valueIndex] = 0  # Set all the invalid values to zero                        
                        if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 1", prevValues, [row], [col], [valueIndex], 1, stepCount=stepCounter):
                            stepCounter += 1
                        
                    # Rules 2 and 3 (2: special case, 1 value in the square. 3: general case)
                    # if N squares in the box contain the same N values and no others, then those values should be removed from 
                    # all other squares in the box. 
                    # This is a brute force approach; I'm not sure there's a better way to do this.

                    squareVals = potentialValues[row, col, :]  # Arbitrarily, only check against the current square
                    prevValues = deepcopy(potentialValues)

                    if np.count_nonzero(squareVals) == 1:  
                        # Special case
                        potentialValues[startRow:startRow+3,startCol:startCol+3,valueIndex] = 0  # Note that this value is taken
                        potentialValues[row, col, valueIndex] = valueIndex + 1  # Re-set this value
                        if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 2", prevValues, [row], [col], [valueIndex], rule=2, stepCount=stepCounter):
                            stepCounter += 1

                    elif np.count_nonzero(squareVals) < 9:
                        # If there are still 9 potential values, there's nothing to check
                        n = np.count_nonzero(squareVals)
                        for squareIndices in combinations(range(9), n):  # Get all possible combinations of n values from [0,8]
                            # The square index is 0-8 and refers to a particular square within a box
                            # Only check this condition for combinations of squares involving the current square
                            if ((row - startRow) * 3 + (col - startCol)) not in squareIndices:
                                continue
                            
                            # Determine the appropriate row and column indices for the squares being checked
                            chosenRows = np.array([(squareIndex // 3) for squareIndex in squareIndices]) + startRow
                            chosenCols = np.array([(squareIndex % 3) for squareIndex in squareIndices]) + startCol
                            
                            # Check that the selected squares only contain n potential values
                            nonzeroCounts = np.array([np.count_nonzero(potentialValues[_r, _c, :]) for _r, _c in zip(chosenRows, chosenCols)])
                            
                            if not np.all(nonzeroCounts == n):
                                continue
                            
                            # Check that all of the chosen squares contain the same values
                            if np.all(np.array([np.all(currSquare == potentialValues[_r, _c, :]) for _r, _c in zip(chosenRows, chosenCols)])):
                                # Remove these values from all other squares in the box
                                for _r in range(3):
                                    for _c in range(3):
                                        if (_r * 3 + _c) not in squareIndices:
                                            potentialValues[startRow+_r, startCol+_c, np.nonzero(currSquare)[0]] = 0

                                if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 3", prevValues, chosenRows, chosenCols, list(range(9)), rule=3, stepCount=stepCounter):
                                    stepCounter += 1


                    # Rule 4
                    # if the intersection of N squares in this box contains N values that *only* appear in
                    # those N squares, then all other values in those squares are invalid
                    prevValues = deepcopy(potentialValues)
                    if 2 <= np.count_nonzero(squareVals):  
                        # This only makes sense if there are at least two potential values
                        # Try to find the largest set in common amongst the squares
                        for n in reversed(range(2, min(8, np.count_nonzero(squareVals)))):  # Iterate over combinations of all sizes from size of (squareVals) down to 2. (note min, not np.min)
                            for squareIndices in combinations(range(9), n):  # Iterate through all possible combinations of n values from [0,8]
                                # Only check if the combination of squares involves the current square (this is an arbitrary choice)
                                if ((row - startRow) * 3 + (col - startCol)) not in squareIndices:
                                    continue
                                
                                # Determine the appropriate row and column indices for the squares being checked
                                chosenRows = np.array([(squareIndex // 3) for squareIndex in squareIndices]) + startRow
                                chosenCols = np.array([(squareIndex % 3) for squareIndex in squareIndices]) + startCol
                                
                                # Find the intersection of the n chosen squares
                                intersection = reduce(np.intersect1d, [potentialValues[_r, _c, :] for _r, _c in zip(chosenRows, chosenCols)])

                                # Get all of the indices *not* in `squareIndices`
                                # Then iterate through them, take any values present in them,
                                # and remove those values from the intersection set
                                intersection = set(intersection)
                                otherSquareIndices = np.setdiff1d(np.arange(9), np.array(squareIndices))  
                                for squareIndex in otherSquareIndices: 
                                    chosenRow = (squareIndex // 3) + startRow
                                    chosenCol = (squareIndex % 3) + startCol
                                    for potentialValue in potentialValues[chosenRow, chosenCol, :]:
                                        if potentialValue in intersection:
                                            intersection.remove(potentialValue)
                            
                                intersection = np.array(list(intersection))  # Cast back to numpy array

                                # Check if the length of the intersection is the same as the number of squares intersected
                                if n == len(intersection):                                        
                                    # That satisfies the constraint! Now, remove all values, except those in the intersection,
                                    # from the intersected squares. Remember that `intersection` contains the actual values,
                                    # but below we want `removeVals` to contain the indices.
                                    removeVals = np.setdiff1d(np.arange(9), intersection - 1)
                                    for _r in range(3):
                                        for _c in range(3):
                                            if (_r * 3 + _c) in squareIndices:
                                                potentialValues[startRow+_r, startCol+_c, removeVals] = 0

                                    if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 4", prevValues, chosenRows, chosenCols, list(range(9)), 4, stepCount=stepCounter):
                                        stepCounter += 1

                    # Rule 5
                    # Row and column checks
                    # If the current value doesn't appear anywhere else in this row or in this column, 
                    # then all other values in this box should be set to zero
                    # Check every row and column except this one
                    if (val not in potentialValues[row, np.arange(9)!=col, :]) and (val not in potentialValues[np.arange(9)!=row, col, :]):
                        prevValues = deepcopy(potentialValues)
                        potentialValues[row, col, np.arange(9)!=valueIndex] = 0  # Set every value except this one to zero

                        if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 5", prevValues, [row], [col], [valueIndex], 5, stepCount=stepCounter):
                            stepCounter += 1


                    # Rule 6
                    # If the current value appears in another box in the same row or column, and in that box
                    # it only appears in that row or column, then it should be set to zero in this box
                    prevValues = deepcopy(potentialValues)
                    # Iterate over the squares in the same row and column as the current square
                    for _i in range(9):
                        # Check if any squares in this row contain val (excluding the current box)
                        if (val in potentialValues[row, _i, :] and not (startCol <= _i < startCol + 3)):
                            # Check if 'val' is the only value in that square.
                            if np.count_nonzero(potentialValues[row, _i, :]) == 1:
                                # If it is, then set the current value to zero
                                potentialValues[row, col, valueIndex] = 0
                                if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 6", prevValues, [row], [_i], [valueIndex], 6, stepCount=stepCounter):
                                    stepCounter += 1
                                break

                            # More generally, check if any other squares in that box, but not in the current row, also contain 'val'
                            foundVal = False
                            boxStartCol = 3 * (_i // 3)
                            for boxRow in range(startRow, startRow+3):
                                if boxRow != row:
                                    if val in potentialValues[boxRow, boxStartCol:boxStartCol+3, :]:
                                        foundVal = True
                                        break

                            # If none of them contain it, then set the current value to zero
                            # Also find out if any other values in the row do contain it, for
                            # the sake of visualization
                            if not foundVal:
                                potentialValues[row, col, valueIndex] = 0
                                
                                containingCols = []
                                for boxCol in range(boxStartCol, boxStartCol + 3):
                                    if val in potentialValues[row, boxCol, :]:
                                        containingCols.append(boxCol)

                                if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 6", prevValues, [row], containingCols, [valueIndex], 6, stepCount=stepCounter):
                                    stepCounter += 1

                                break


                        # Check if any squares in this column contain val (excluding the current box)
                        if (val in potentialValues[_i, col, :] and not (startRow <= _i < startRow + 3)):
                            # Check if 'val' is the only value in that square.
                            if np.count_nonzero(potentialValues[_i, col, :]) == 1:
                                # If it is, then set the current value to zero
                                potentialValues[row, col, valueIndex] = 0
                                if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 6", prevValues, [_i], [col], [valueIndex], 6, stepCount=stepCounter):
                                    stepCounter += 1
                                break

                            # More generally, check if any other squares in that box, but not in the current column, also contain 'val'
                            foundVal = False
                            boxStartRow = 3 * (_i // 3)
                            for boxCol in range(startCol, startCol+3):
                                if boxCol != col:
                                    if val in potentialValues[boxStartRow:boxStartRow+3, boxCol, :]:
                                        foundVal = True
                                        break

                            # If none of them contain it, then set the current value to zero
                            # Also find out if any other values in the column do contain it, for
                            # the sake of visualization
                            if not foundVal:
                                potentialValues[row, col, valueIndex] = 0
                                
                                containingRows = []
                                for boxRow in range(boxStartRow, boxStartRow + 3):
                                    if val in potentialValues[boxRow, col, :]:
                                        containingRows.append(boxRow)
                                
                                if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 6", prevValues, containingRows, [col], [valueIndex], 6, stepCount=stepCounter):
                                    stepCounter += 1
                                break


                    # If the above rules failed me, I would normally start guessing and see if the guesses pan out... but I ran
                    # across a Numberphile video that might provide a better approach (and be easier to implement)
                    # Phistomephel Ring https://www.youtube.com/watch?v=pezlnN4X52g
                    # The set of numbers in the squares ringing the center box and the set of numbers
                    # in the four groups of four corner boxes are identical.     
                    # First, get all of the ring values
                    ringIndices = [(2,2), (2,3), (2,4), (2,5), (2,6), 
                                   (3,2), (4,2), (5,2), 
                                   (3,6), (4,6), (5,6), 
                                   (6,2), (6,3), (6,4), (6,5), (6,6)]
                    ringSet = set([])
                    for indices in ringIndices:
                        ringSet.update(potentialValues[indices[0], indices[1], :])

                    # Then, get all of the corner values
                    cornerIndices = [(0,0), (0,1), (1,0), (1,1), 
                                   (7,7), (7,8), (8,7), (8,8), 
                                   (0,7), (0,8), (1,7), (1,8), 
                                   (7,0), (8,0), (7,1), (8,1)]
                    cornerSet = set([])
                    for indices in cornerIndices:
                        cornerSet.update(potentialValues[indices[0], indices[1], :])

                    # Find the intersection of the two sets
                    phistomephelSet = ringSet.intersection(cornerSet)

                    # Remove any values not contained in the intersection
                    prevValues = deepcopy(potentialValues)
                    for indices in ringIndices:
                        for i in range(9):
                            if potentialValues[indices[0], indices[1], i] not in phistomephelSet:
                                potentialValues[indices[0], indices[1], i] = 0

                    for indices in cornerIndices:
                        for i in range(9):
                            if potentialValues[indices[0], indices[1], i] not in phistomephelSet:
                                potentialValues[indices[0], indices[1], i] = 0
                    
                    currRows = np.concatenate((np.array(cornerIndices)[:,0], np.array(ringIndices)[:,0]))
                    currCols = np.concatenate((np.array(cornerIndices)[:,1], np.array(ringIndices)[:,1]))
                    if not prettyPlot(potentialValues, f"Step: {stepCounter}, Rule 7", prevValues, currRows=currRows, currCols=currCols, currValIndexes=np.arange(9), rule=7, stepCount=stepCounter):
                        stepCounter += 1


        # Check if we've gotten stuck (looped through the entire sudoku with no changes)
        if prevStepCounter != stepCounter:
            # Not stuck yet
            prevStepCounter = stepCounter
            #prettyPlot(potentialValues, f"One more loop down. {stepCounter} steps.", stepCount=stepCounter)

        else:
            #print("We're stuck. Time to guess!")

            # Save off the current state,
            potentialValuesCopy = deepcopy(potentialValues)
            
            # Find a square with the fewest number of possible values remaining,
            nonzero_counts = np.sum(potentialValues != 0, axis=2)  # Count nonzero values in each square
            if np.any(nonzero_counts == 0):  # If any cells are all zero, there's no solution
                return False
            nonzero_counts[nonzero_counts==1] = 9  # We aren't interested in solved squares (1 nonzero value), so change them to a high number
            min_count = np.min(nonzero_counts)  # Find the smallest number of nonzero values
            min_indices = np.argwhere(nonzero_counts == min_count)  # Find the indices of the minimum nonzero count

            # Choose the first of these squares with the minimum number of potential values
            chosenSquare = min_indices[0]

            #print("Guess square: ", potentialValues[chosenSquare[0], chosenSquare[1], :])
            # Find the first nonzero value from that square, then set all the others to zero
            for chosenValIndex in range(9):
                if potentialValues[chosenSquare[0], chosenSquare[1], chosenValIndex] != 0:
                    break
            
            #print("Guess: ", potentialValues[chosenSquare[0], chosenSquare[1], chosenValIndex])
            potentialValues[chosenSquare[0], chosenSquare[1], chosenValIndex+1:] = 0
            prettyPlot(potentialValues, f"Step {stepCounter}, Recursion depth {recursionDepth+1}.", potentialValuesCopy, [chosenSquare[0]], [chosenSquare[1]], [chosenValIndex], 8)

            # Run the solver with this new sudoku. It will return None if the Sudoku is unsolvable.
            returnedSudoku = solver1(potentialValues, stepCounter, recursionDepth+1, itr)
            if returnedSudoku is None:
                #print("Guess was wrong. Moving on.")
                # It failed, so the value that we just selected is wrong. Re-set the potentialValues, but zero that one.
                # Then, having figured out one more incorrect value, continue on with solving as usual
                potentialValuesCopy[chosenSquare[0], chosenSquare[1], chosenValIndex] = 0
                potentialValues = deepcopy(potentialValuesCopy)

            else:
                # If we get a success returned here, we return it up the stack, or back to the original call
                return returnedSudoku    
            

        # Check if the solution is bad (i.e. one of the squares no longer contains any potential values)
        if np.any(np.all(potentialValues[:, :, :] == 0, axis=2)):
            if recursionDepth == 0:
                print(f"Failed! In {stepCounter} steps.")
                prettyPlot(potentialValues, f"Failed! In {stepCounter} steps.")
            return None

        # Check if it's solved
        solved = True
        for row in range(9):
            for col in range(9):
                if np.count_nonzero(potentialValues[row, col]) > 1:
                    solved = False
                    break
            if not solved:
                break

    if solved:
        print(f"Solved! In {stepCounter} steps.")
        pass#prettyPlot(potentialValues, f"Solved! In {stepCounter} steps.")
    else:
        print(f"Failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Return the completed sudoku
    return np.max(potentialValues, axis=2)
                    

def solver2(_sudoku):
    # This should be the fully recursive solver

    # Track the unused numbers in each row, column, and box
    # Find locations where there is only one available number 
    # between a cell's row, column, and box.

    # TODO: I think this isn't working because I don't account for the 
    #       "row is constrained even though a value hasn't been set" case.
    #       I think I need sub-column and row sets, and I think if I have those,
    #       then I don't need the overall column/row sets.

    rowNums = []  # Track available numbers for the rows
    colNums = []  # Track available numbers for the columns
    boxNums = []  # Track available numbers for the boxes
    for col in range(9):
        rowNums.append(set({1,2,3,4,5,6,7,8,9}))
        colNums.append(set({1,2,3,4,5,6,7,8,9}))
        boxNums.append(set({1,2,3,4,5,6,7,8,9}))

    # Update the sets with the initial numbers
    for i in range(9):
        # Handle each row
        nonZeroRowNums = _sudoku[i,:][_sudoku[i,:]!=0]
        rowNums[i] = rowNums[i] - set(nonZeroRowNums)

        # Handle each column
        nonZeroColNums = _sudoku[:,i][_sudoku[:,i]!=0]
        colNums[i] = colNums[i] - set(nonZeroColNums)

        # Handle each box (each 3x3)
        startRow = 3 * (i // 3)
        startCol = 3 * (i % 3)
        nonZeroBoxNums = _sudoku[startRow:startRow+3,startCol:startCol+3][_sudoku[startRow:startRow+3,startCol:startCol+3]!=0]
        boxNums[i] = boxNums[i] - set(nonZeroBoxNums)

    # Check every cell to see if it's fully constrained. If so, populate it and update the sets.
    counter = 0
    prettyPlot(_sudoku)
    while 0 in _sudoku and counter < 100:
        counter += 1
        for row in range(9):
            for col in range(9):
                box = 3 * (row // 3) + (col // 3)
                inter = rowNums[row].intersection(colNums[col], boxNums[box])
                if len(inter) == 1:
                    num = inter.pop()
                    _sudoku[row,col] = num
                    rowNums[row].remove(num)
                    colNums[col].remove(num)
                    boxNums[box].remove(num)
                    prettyPlot(_sudoku, rowNums, colNums, boxNums, (row,col))
    
    print(f"Counter: {counter}")
    print(_sudoku)


def solver3(_sudoku):
    """
    We can set up the sudoku as an optimization problem.
    Each row, column, and box must contain the whole numbers
    1 - 9, and each cannot contain duplicates.
    I will use pyomo to set up and solve this problem.
    """
    # Create a model
    model = ConcreteModel()

    # Sets
    rows = RangeSet(1, 9)
    cols = RangeSet(1, 9)

    # Set up the problem: a 9x9 grid of integers limited to the integers [1,9]
    model.x = Var(rows, cols, within=PositiveIntegers, bounds=(1,9))

    # Row uniqueness constraint
    #TODO: Neither of these constraints uses correct syntax. Maybe we forget the constraint, just optimize to the Sum[1,9] objectives?
    model.row_constraint = Constraint(rows, cols, cols, rule=lambda model, i, j, k: (j != k).implies(model.x[i, j] != model.x[i, k]))
    model.row_constraint = Constraint(rows, cols, cols, rule=lambda model, i, j, k: (j != k) == (model.x[i, j] != model.x[i, k]))





    # Each value must appear once in each row
    model.row_constraint = Constraint(rows, rule=lambda model, i: sum(model.x[i, j] for j in cols) == 1)

    # Each value must appear once in each column
    model.col_constraint = Constraint(Cols, Values, rule=lambda model, j, v: sum(model.x[i, j, v] for i in Rows) == 1)

    

    # The value each row, column, and box should sum to
    maxVal = np.sum(np.arange(10))

    # Objective Function. Minimize the difference between each row, column, and box sum and maxVal


    return model


def solver4(_sudoku):
    """
    Can we solve a sudoku with linear back projection?
    """


startTime = time.time()
for i, _sudoku in enumerate(sudoku):
    print(f"Sudoku {i}")
    solver1(_sudoku)

print(f"Solver 1 solved all 50 Sudoku in {time.time() - startTime} seconds.")
