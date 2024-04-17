# Sudoku Solver  
  
In the last 19 years I've played through XX of the Sudoku in the Nintendo DS game Brain Age.  
I have YY Sudoku remaining.  
  
So, I decided it was time to write a Sudoku solver. For the sake of the exercise, I started this project without looking into best practices, the current state of the art, or the hundreds of existing Sudoku solver packages, open source projects, and college homework assignments that must exist. In the end I wrote six different solvers. Two of them work, two of them do not work, and two of them might work if they were able to run for a really, really long time. The solvers are described in more detail below.  
  
SudokuSolver.py contains the six solvers, a function to draw out the Sudoku (complete, incomplete, and in-progress), a function to check a Sudoku, and code to read in and format the Sudoku in the text file sudoku.txt  
sudoku.txt contains 50 Sudoku puzzles. I copied this file from from [dimitri] https://github.com/dimitri/sudoku/tree/master  
  
Below are descriptions of each solver and details about how they work. Solvers 1 and 4 work, solvers 2 and 3 might work, if given enough time, solvers 5 and 6 do not work. First, the rules  

##### Sudoku Rules  
A 9×9 square must be filled in with numbers from 1-9 with no repeated numbers in each line, horizontally or vertically. To challenge you more, there are 3×3 squares marked out in the grid, and each of these squares can't have any repeat numbers either.  
From [Senior Lifestyle](https://www.seniorlifestyle.com/resources/blog/5-tips-sudoku-beginners/#:~:text=The%20rules%20for%20sudoku%20are,have%20any%20repeat%20numbers%20either.)  
  
##### Solver 1: "By Hand"  
This is approximately how I would solve a Sudoku by hand. This is the longest solver in terms of code and its description, and despite being second nature by hand, it was deceptively hard to translate into code.  
This is also a slow solver. I haven't made any attempt to profile it, but it solved all 50 Sudoku in 834 seconds, or 16.7 seconds per Sudoku.  
  
I solve Sudoku by a process of elimination. At the start, I assume each square can contain any of the numbers 1 - 9. Then, I go through the squares and eliminate all numbers which are not valid. Eventually, I'm left with one number in each square. If I reach a point where I can't seem to eliminate any more numbers, but the Sudoku still isn't solved, I will guess, and if I'm wrong, back track. I assume that in such a case, I'm missing something, and I'm just "bad at Sudoku." (The quotes aren't necessary, but I think they're funny).  
I eliminate numbers through six rules. In the process of making this, I ran across a Numberphile video on Sudoku (The Phistomefel Ring), and added it as an additional rule, despite never using it myself.  
  
The seven rules are, in no particular order  
    
###### Rule 1  
Considering one possible value. If the current value doesn't appear anywhere else in this box, then all other values in this square are invalid. ('box' referring to each of the nine 3x3 boxes)    
  
###### Rule 2  
If there is only one possible value for a given square, then that value is not a possibility for the other squares in that box. (i.e. if one of the squares has a set value, remove that value from the other squares in the box)     
  
###### Rule 3  
If N squares in the box contain the same N potential values and no other potential values, then those values should be removed from all other squares in the box. In other words, each of these N squares must contain one of these N values, so the values couldn't possibly be in other squares in this box.   
(This is the general case of Rule 2)  

###### Rule 4  
If the intersection of N squares in a box is N values which *only* appear in those N squares, then all other values in those squares are invalid. In other words, each of these N squares must contain one of those N values, so those squares couldn't possibly contain other values.  
  
###### Rule 5  
If one potential value doesn't appear anywhere else in its row or column, then all other values in its square should be set to zero. (i.e. that potential value is the answer for that square)  
  
###### Rule 6  
If a potential value in one square appears in another box, in the same row or column, and in that box it only appears in the same row or column as the value being considered, then the value being considered is invalid.  
  
###### Rule 7  
The [Phistomefel Ring](https://www.youtube.com/watch?v=pezlnN4X52g)  
The set of numbers in the squares ringing the center box and the set of numbers in the four groups of four corner boxes are identical.  
  
And the informal Rule 8: guess. Guessing is done using recursion: a guess is made by selecting a value for a square. The resulting values are passed, recursively, to solver 1. If solver 1 fails, it will return the failure up the stack, and a different guess will be made. If the guess is correct, the solved Sudoku will be returned up the stack, back to the original call.  
    
##### Solver 2: "Random Guessing"  
##### Solver 3: "Slightly Better Random Guessing"  
##### Solver 4: "Integer Programming"  
##### Solver 5: "Algebraic Solver"  
##### Solver 6: "Iterative Linear Back Projection"  
