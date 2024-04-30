# Sudoku Solver  
   
I've played through 53 of the 120 Sudoku in the Nintendo DS game Brain Age.  
It's been 19 years.  
So, I decided it was time to write a Sudoku solver. For the sake of the exercise, I started this project without looking into best practices, the current state of the art, or the hundreds of existing Sudoku solver packages, open source projects, and college homework assignments that must exist. In the end I wrote six different solvers. Two of them work, two of them do not work, and two of them might work if they were able to run for a really, really long time. The solvers are described in more detail below.  
  
SudokuSolver.py contains the six solvers, a function to draw out the Sudoku (complete, incomplete, and in-progress), a function to check a Sudoku, and code to read in and format the Sudoku in the text file sudoku.txt  
sudoku.txt contains 50 Sudoku puzzles. I copied this file from from [dimitri] https://github.com/dimitri/sudoku/tree/master  
  
Below are descriptions of each solver and details about how they work. Solvers 1 and 4 work, solvers 2 and 3 work, but take an extremely long time to find a solution on all but the easiest Sudoku, solvers 5 and 6 do not work. First, the rules  

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
This solver randomly places the numbers missing from each row, then checks for validity. If it's invalid, it tries again.  
Incorrect guesses are cached so they aren't guessed again; this saves the overhead of re-checking the guess.  
  
This is a terrible way to solve a Sudoku, but it does work. Consider that a uniquely solvable Sudoku has at least 17 given values; let's assume a puzzle with 18 given values for convenience. Assume there are 2 given values per row. There are 7P7 ways to place the unknown numbers in each of the 9 rows, so there are (7P7)^9 = 2.1 x 10^33 potential guesses.   
  
(Sudoku 52 was made as a very, very easy version of Sudoku 1, so this solver could find a solution in a reasonable amount of time)  
  
##### Solver 3: "Slightly Better Random Guessing"  
This solver randomly places the numbers missing from each row, one row at a time, and checks for validity after each guess. My rough upper bound puts the number of possible guesses at 1.8 x 10^20. That's a huge improvement! It still isn't reasonable, though.    
The details of how I got to that number are in the comment block for Solver 3, in the code. It's an interesting problem, and I couldn't find a solution out there. Let me know if you know of one!  
  
(Sudoku 51 was made as a very easy version of Sudoku 1, so this solver could find a solution in a reasonable amount of time)  
  
##### Solver 4: "Integer Programming"  
My passing knowledge of Sudoku solvers tells me that integer programming is the "correct" way to solve a Sudoku in code. (Don't take my word for it, though!)  
The premise is straightforward. We set up an optimization problem with the basic Sudoku rules  
1. There are 81 numbers in a Sudoku. These numbers map to a 9x9 grid with 9 rows, 9 columns, and 9 3x3 boxes.  
2. Each number can only be one of the integers 1 - 9.  
3. Each number 1 - 9 can only appear once in each row  
4. ... in each column  
5. ... and in each box  
6. Some of these values are already given.  
  
Then, allow a solver to find the optimal solution. This approach really pushes off the hard work onto the solver; in this case, the GLPK solver, used through the Pyomo package.  
  
##### Solver 5: "Algebraic Solver"  
This approach sets up a series of equations such that each row, column, and box in the Sudoku must add to the sum of the numbers 1 - 9, and their products must be equal to the product of the numbers 1 - 9. There are three reasons why this doesn't work  
1. The GLPK solver used in Solver 4 allows for an integer-only constraint, but does not work with equations involving multiplication, and the IPOPT nonlinear optimization package doesn't allow for an integer-only constraint. (I used IPOPT and crossed my fingers.)  
2. IPOPT doesn't work if there are more constraints than unknowns. We have 9 * 6 = 54 constraints for sums and products, so there are often more constraints than unknowns, and the solver fails outright. (This also means that adding our own integer constraint exacerbates this problem.)  
3. [It turns out these constraints aren't sufficient.](https://hkopp.github.io/2021/08/solving-sudoku-algebraically)  
  
_It's entirely possible that 1 and 2 aren't insurpassable problems, and are only a result of my limited knowledge of GLPK, IPOPT, and other available packages, and my decision to not write my own solver._  
  
##### Solver 6: "Iterative Linear Back Projection"   
As of today (April 21, 2024), [Wikipedia has a great video demonstrating linear back projection (LBP)](https://en.wikipedia.org/wiki/Tomographic_reconstruction). LBP is a general concept and can be applied broadly, but it is widely known as a basic way to reconstruct X-Ray CT scans. Imagine an X-Ray machine rotates around a person. It takes measurements of a 2D slice of their body from many angles. At each angle the measurement only shows the amount of signal that passed through the body. Now, flip this scenario. At each angle, project a 'ray' that is scaled by the measurement. Do this at all angles, and sum together the overlapping rays. The result is an image of the subject.  
  
Following this approach, we know the "measurement" of each row, column, and box of a Sudoku should be 45 - i.e. the sum of the values 1 - 9. We take each of these sums and subtract from it the relevant values that are given to us. The remaining value is divided by the number of unknown squares in the row, column, or box, and is "projected" into the squares. The values projected into the squares are summed, and divided by three (because there are three projections). The values can then be clamped, either to the range 1 - 9, or such that they are integers.  
  
The result of LBP may not be correct - i.e. if we simulate the measurement process (summing rows, columns, and boxes) they might not add to the "measured" values, so we can repeat the LBP process by projecting the errors onto the produced "image." This repeated process makes it iterative linear back projection (ILBP).  
  
As mentioned in Solver 5, the sum and product constraints are not sufficient to constrain a Sudoku, so there's no reason to think the sum by itself would constrain it - which is essentially assumed here.  
  