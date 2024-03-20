# Homework
#=====================================

left_number = float( input() )
right_number = float( input() )

character = input()

if character == "/":
    if right_number == 0 :
        print("Input Illegal!!!")
    else :
        print( left_number / right_number )
        
elif character == "+" :
    print( left_number + right_number )

elif character == "-" :
    print( left_number - right_number )

elif character == "*" :
    print( left_number * right_number )

elif character == "**" :
    print( left_number ** right_number )

else :
    print("Input Illegal!!!")
    
#=====================================