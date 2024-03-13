money = int(input("Please input the initial fund:"))
month = int(input("Please input the time of duration:"))
rate = int(input("Please input the rate of per month:"))

ans = money * ( ( 1+rate ) ** month )

print(ans - money)