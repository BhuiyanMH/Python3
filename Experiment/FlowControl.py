# for val in sequence:
#     Body of for

numbers = [6, 5, 3, 4, 11]
sum = 0
for val in numbers:
    sum += val
print("The sum is ", sum)

#Range function: we can generate all numbers using range function
#range(start, stop, stepSize)

print(list(range(5)))
print(list(range(5,10)))

for value in range(2,5):
    print(value)

#Range can also be used with len function
genre = ['pop', 'rock', 'jazz']

for i in range(len(genre)):
    print("I like ", genre[i])

# For with else: else part is executed if the items
# in the sequence used in for loop exhausts
#if break is used, else part is ignored

digits = [0, 1, 2, 3, 4, 5]
inputData = int(input("Enter an integer: "))

for d in digits:
    if d == inputData:
        print("Digit is in the list")
        break
else:
    print("Digit is not in the list")

#Python while loop: while can have a optional else like for
#else executes if the condition in the while loop evaluates to false
# while test_expression:
#     Body of while

