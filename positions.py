import numpy as np

class openPosition():
    def __init__(self, price: float, units: float, time: int, type: str, 
                 rollOverPercentage: float, fee: float, rollOverEvery: int = 4):
        self.openPrice      = price
        self.units          = units
        self.openAt         = time
        self.type           = type
        self.openCost       = units * price * (1.0+fee)
        self.lastRollOver   = time
        self.rollOverPeriod = rollOverEvery
        self.rollOverPerc   = rollOverPercentage
        self.rollOverCost   = 0.0
        self.transactionFee = fee
        self.lossMargin     = -5.0 if (type == 'long') else +5.0
        self.earnMargin     = +5.0 if (type == 'long') else -5.0
        self.margin         = np.NaN

    def evaluatePosition(self,currentPrice,time):
        if time - self.lastRollOver > self.rollOverPeriod:
            self.rollOverCost += self.rollOverPerc * currentPrice * self.units
            self.lastRollOver = time

        totalPrice = currentPrice*self.units
        totalCost  = self.openCost + self.rollOverCost * (1.0 if self.type == 'long' else -1.0)

        if self.type == 'long': 
            if (totalPrice - (totalCost + totalPrice*self.transactionFee) > self.earnMargin) or \
               (totalPrice - (totalCost + totalPrice*self.transactionFee) < self.lossMargin):
                self.margin = totalPrice*(1.0-self.transactionFee) - totalCost
        else:
            if (totalCost - totalPrice*(1.0+self.transactionFee)> self.earnMargin) or \
               (totalPrice*(1.0+self.transactionFee) - totalCost> self.lossMargin):
                self.margin = totalCost - totalPrice*(1.0+self.transactionFee)
            
        return self.margin

def checkOpenPositions(openPositions,historicPositions,currentTime,currentPrice):
    for i, pos in enumerate(openPositions):
        exchange = pos.evaluatePosition(currentPrice,currentTime)

        if exchange:
            totalExchange += exchange
            historicPositions.append(pos)
            openPositions.pop(i)