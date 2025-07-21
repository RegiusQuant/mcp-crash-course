from fastmcp import FastMCP


mcp = FastMCP("Flight Booker")


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    Search for available flights.

    Parameters:
      - origin (str): 3-letter IATA code of departure airport (e.g. "JFK").
      - destination (str): 3-letter IATA code of arrival airport (e.g. "LAX").
      - date (str): Travel date in YYYY-MM-DD format (e.g. "2025-06-15").

    Returns:
      A newline-separated list of flights, each in the exact format:
        FlightID: <ID>, Airline: <Name>, Departs: <HH:MM>, Arrives: <HH:MM>, Price: $<amount>
      Example:
        FlightID: AB1234, Airline: Acme Air, Departs: 09:30, Arrives: 12:45, Price: $320
    """
    # Mock data
    return "\n".join([
        "FlightID: AB1234, Airline: Acme Air, Departs: 09:30, Arrives: 12:45, Price: $320",
        "FlightID: XY5678, Airline: SkyHigh, Departs: 14:00, Arrives: 17:15, Price: $290",
    ])


@mcp.tool()
def book_flight(flight_id: str, passenger_name: str) -> str:
    """
    Book a flight with the specified flight ID.

    Parameters:
      - flight_id (str): The exact FlightID from search_flights output (e.g. "AB1234").
      - passenger_name (str): Full name of the passenger (e.g. "Alice Smith").

    Returns:
      A confirmation string in the exact format:
        BookingID: <UUID>, FlightID: <same as input>, Passenger: <name>, Status: CONFIRMED
      Example:
        BookingID: 3f47a2de-9b7c-4b8e-8c1e-123456abcdef, FlightID: AB1234, Passenger: Alice Smith, Status: CONFIRMED
    """
    # Mock booking logic (would generate a real UUID in production)
    booking_id = "3f47a2de-9b7c-4b8e-8c1e-123456abcdef"
    return f"BookingID: {booking_id}, FlightID: {flight_id}, " f"Passenger: {passenger_name}, Status: CONFIRMED"


@mcp.tool()
def send_confirmation(booking_id: str, email: str) -> str:
    """
    Send the booking confirmation to the passenger’s email.

    Parameters:
      - booking_id (str): The BookingID from book_flight output (e.g. "3f47a2de-…").
      - email (str): Passenger’s email address (e.g. "alice@example.com").

    Returns:
      A success message confirming dispatch:
        Confirmation sent for BookingID <booking_id> to <email>
      Example:
        Confirmation sent for BookingID 3f47a2de-… to alice@example.com
    """
    # Mock email send
    return f"Confirmation sent for BookingID {booking_id} to {email}"


if __name__ == "__main__":
    mcp.run()