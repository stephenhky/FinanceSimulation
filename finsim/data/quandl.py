
import quandl


class QuandlReader:
    def __init__(self, token):
        self.token = token

    def get_treasury_real_longterm_rates(self):
        return quandl.get("USTREASURY/REALLONGTERM", authtoken=self.token)

    def get_treasury_real_yield_curve_rates(self):
        return quandl.get("USTREASURY/REALYIELD", authtoken=self.token)

    def get_treasury_yield_curve_rates(self):
        return quandl.get("USTREASURY/YIELD", authtoken=self.token)

    def get_treasury_longterm_rates(self):
        return quandl.get("USTREASURY/LONGTERMRATES", authtoken=self.token)
