# Practice In Class
#===================================

score = int( input() )

if score < 101 and score > -1:
    
    if score > 89 :
        print("Excellent")
    
    elif score > 79 :
        print("Good")
        
    elif score > 69 :
        print("Medium")
        
    elif score > 59 :
        print("Passed")
        
    else :
        print("Failed")

else :
    print("Wrong Score")
    
#=====================================