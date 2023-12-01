# Sudoku Solver
#
# Reads the file "sudoku.txt" (from [dimitri] https://github.com/dimitri/sudoku/tree/master)
# and solves them.
#
# Daniel Winker, November 30, 2023
# TODO: Solve through optimization.
# TODO: Try solving with linear back projection. I project back sum of the unused row, column, and box values. Normalize. Scale 1 - 9.
# TODO: Could Loopy BP or something similar be used here?

import time
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from itertools import combinations
from functools import reduce

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


def prettyPlot(_sudoku, _rowNums=[[set({})]]*9, _colNums=[[set({})]]*9, _boxNums=[[set({})]]*9, lastChange=None, scale=2):
    """
    This creates a visual to understand how the solver is working.
    Written with ChatGPT.
    """

    fig, ax = plt.subplots()

    # ___Create the Sudoku grid and row, column, and box visuals___

    half = 9 * scale / 2  # Subtract this from things to center it

    # Remove grid lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Remove the outer box
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid
    for i in range(0, 10):
        if i % 3 == 0:
            linewidth = 2  # use a thicker line for the outer lines
        else:
            linewidth = 1  # use a thinner line for the inner lines

        plt.plot([0,9*scale],[i*scale,i*scale], color='black', linewidth=linewidth)
        plt.plot([i*scale,i*scale], [0,9*scale], color='black', linewidth=linewidth)

    # For the rows
    TODO

    # For the columns
    TODO 

    # For the boxes
    TODO

    # Set axis limits
    ax.set_xlim(-0.5, 9*scale+0.5)
    ax.set_ylim(-0.5, 9*scale+0.5)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')

    # ___Populate it with numbers___

    # The Sudoku
    for row in range(9):
        for col in range(9):
            num = _sudoku[row,col]
            if num != 0:
                if num - int(num) == 0:
                    num = int(num)
                plt.text((8 - col +0.5)*scale, (8 - row +0.5)*scale, num, ha='center', va='center')  # x-coordinate, y-coordinate, text

    # The rows

    # The columns

    # The boxes


    plt.show()


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

def solver1(_sudoku):
    """
    The naive, brute force way to solve this is
    1. Assume any empty square can hold a number 1 - 9
    2. Select one square
    3. Check the box, row, and column to see if any of the values in the square are invalid
    4. Repeat until one valid value remains in each square
    """

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
    
    # Iterate over every square until each square only contains one nonzero value
    solved = False
    while not solved:
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


                    # Check the values in the box
                    # If the current value in this square doesn't appear anywhere else
                    # in this box, then all other values in this square should be assigned 0
                    if val not in currBox[np.arange(9)!=row, np.arange(9)!=col, :]:
                        potentialValues[row, col, np.arange(9)!=valueIndex] = 0  # Set all the invalid values to zero                        
                    
                    # if N squares contain the same N values and no others, then those values should be removed from 
                    # all other squares in the box. 
                    # This is a brute force approach; I'm not sure there's a better way to do this.
                    mainSet = potentialValues[row, col, :]  # Arbitrarily, only check against the current square
                    if np.count_nonzero(mainSet) == 1:  # Special case
                        potentialValues[startRow:startRow+3,startCol:startCol+3,valueIndex] = 0  # Note that this value is taken
                        potentialValues[row, col, valueIndex] = valueIndex + 1  # Re-set this value

                    elif np.count_nonzero(mainSet) < 9:  # If there are still 9 potential values, there's nothing to check
                        n = np.count_nonzero(mainSet)
                        for squareIndices in combinations(range(9), n):  # Get all possible combinations of n values from [0,8]
                            # Only check this condition for combinations of squares involving the current square
                            if (row * 3 + col) not in squareIndices:
                                continue
                            
                            # Determine the appropriate row and column indices for the squares being checked
                            chosenRows = [(squareIndex // 3) for squareIndex in squareIndices]
                            chosenCols = [(squareIndex % 3) for squareIndex in squareIndices]

                            # Check that the selected squares only contain n potential values
                            nonzeroCounts = [np.count_nonzero(currBox[_r, _c, :]) for _r, _c in zip(chosenRows, chosenCols)]
                            if not np.all(nonzeroCounts == n):
                                continue

                            # Check that all of the chosen squares contain the same values
                            if np.all([np.all(currSquare == currBox[_r, _c, :]) for _r, _c in zip(chosenRows, chosenCols)]):
                                # Remove these values from all other squares in the box
                                for _r in range(startRow,startRow+3):
                                    for _c in range(startCol,startCol+3):
                                        if (row * 3 + col) not in squareIndices:
                                            potentialValues[_r, _c, np.nonzero(currSquare[row,col,:])[0]-1] = 0


                    # if the intersection of N squares in this box is N values, and those N values *only* appear in
                    # those N squares, then all other values in those squares are invalid
                    if 2 <= np.count_nonzero(mainSet):  # This only makes sense if there are at least two potential values
                        # Try to find the largest set in common amongst the squares
                        for n in reversed(range(2, np.min(8, np.count_nonzero(mainSet)))):  # Iterate over combinations of all sizes from size of (mainset) down to 2
                            for squareIndices in combinations(range(9), n):  # Get all possible combinations of n values from [0,8]
                                # Only check if the combination of squares involves the current square (this is an arbitrary choice)
                                if (row * 3 + col) not in squareIndices:
                                    continue

                                # Determine the appropriate row and column indices for the squares being checked
                                chosenRows = [(squareIndex // 3) for squareIndex in squareIndices]
                                chosenCols = [(squareIndex % 3) for squareIndex in squareIndices]
                                
                                # Find the intersection of the chosen squares
                                intersection = reduce(np.intersect1d(currBox[chosenRows, chosenCols, :]))

                                # Check if the length of the intersection is the same as the number of squares intersected
                                if n == len(intersection):
                                    # Get the indices of all squares not involved in this intersection
                                    otherSquareIndices = np.setdiff1d(np.arange(9), np.array(squareIndices))
                                        
                                    # Determine the appropriate row and column indices for the squares being checked
                                    otherRows = [(otherSquareIndex // 3) for otherSquareIndex in otherSquareIndices]
                                    otherCols = [(otherSquareIndex % 3) for otherSquareIndex in otherSquareIndices]

                                    # Check that the values in the intersection don't appear anywhere else in the box
                                    if not np.any(intersection == currBox[_r, _c, :] for _r, _c in zip(otherRows, otherCols)):
                                        # That satisfies the constraint! Now, remove all values except those in the intersection
                                        # from the intersected squares
                                        for _r in range(startRow,startRow+3):
                                            for _c in range(startCol,startCol+3):
                                                if (row * 3 + col) in squareIndices:
                                                    potentialValues[_r, _c, np.nonzero(intersection)[0]-1] = 0


                    # Row and column checks
                    # If the current value doesn't appear anywhere else in this row or in this column, 
                    # then all other values in this box should be set to zero
                    # Check every row and column except this one
                    if (val not in potentialValues[row, np.arange(9)!=col, :]) or (val not in potentialValues[np.arange(9)!=row, col, :]):
                        potentialValues[row, col, np.arange(9)!=valueIndex] = 0  # Set every value except this one to zero

                    # If the current value appears elsewhere in this row or column, in a different box, and 
                    # it's the only nonzero value in that box, then it should be set to zero in this box
                    for _i in range(9):
                        # This if statement reads, "if there's only one nonzero value in the selected square, and that value is `val`,
                        #                           and that square is not the current square"
                        if (np.count_nonzero(potentialValues[row, _i, :]) == 1 and np.nonzero(potentialValues[row, _i, :])[0][0] == val or \
                            np.count_nonzero(potentialValues[_i, col, :]) == 1 and np.nonzero(potentialValues[_i, col, :])[0][0] == val) and \
                           _i != col and _i != row:
                             potentialValues[row, col, val-1] = 0
                    

def solver2(_sudoku):
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
    TODO: Neither of these constraints uses correct syntax. Maybe we forget the constraint, just optimize to the Sum[1,9] objectives?
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


print(sudoku[0])
solver1(sudoku[0])
