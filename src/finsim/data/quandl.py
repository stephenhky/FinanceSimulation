
import pandas as pd
import quandl


class QuandlReader:
    """A class to read financial data from Quandl."""
    
    def __init__(self, token: str):
        """Initialize the QuandlReader with an API token.
        
        Args:
            token: The API token for Quandl
        """
        self.token = token

    def get_treasury_real_longterm_rates(self) -> pd.DataFrame:
        """Get Treasury real long-term rates.
        
        Returns:
            A DataFrame containing Treasury real long-term rates
        """
        return quandl.get("USTREASURY/REALLONGTERM", authtoken=self.token)

    def get_treasury_real_yield_curve_rates(self) -> pd.DataFrame:
        """Get Treasury real yield curve rates.
        
        Returns:
            A DataFrame containing Treasury real yield curve rates
        """
        return quandl.get("USTREASURY/REALYIELD", authtoken=self.token)

    def get_treasury_yield_curve_rates(self) -> pd.DataFrame:
        """Get Treasury yield curve rates.
        
        Returns:
            A DataFrame containing Treasury yield curve rates
        """
        return quandl.get("USTREASURY/YIELD", authtoken=self.token)

    def get_treasury_longterm_rates(self) -> pd.DataFrame:
        """Get Treasury long-term rates.
        
        Returns:
            A DataFrame containing Treasury long-term rates
        """
        return quandl.get("USTREASURY/LONGTERMRATES", authtoken=self.token)
