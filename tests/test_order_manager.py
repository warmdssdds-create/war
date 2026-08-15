from __future__ import annotations

from src.exchange.order_manager import OrderIntent, OrderManager, generate_order_link_id


class DummyRest:
    def __init__(self) -> None:
        self.cancel_called = False
        self.stop_updates: list[tuple[int, float | None, float | None]] = []

    def place_market_order(self, **kwargs):
        return {"result": {"orderId": "abc123", "orderStatus": "New", "cumExecQty": 0.0, **kwargs}}

    def cancel_all_orders(self, **kwargs):
        self.cancel_called = True
        return {"result": kwargs}

    def set_trading_stop(self, category, symbol, position_idx, stop_loss, take_profit, trailing_stop):
        self.stop_updates.append((position_idx, stop_loss, take_profit))
        return {"result": {"category": category, "symbol": symbol, "position_idx": position_idx}}



def test_order_link_id_is_idempotent() -> None:
    intent = OrderIntent(symbol="ETCUSDT", side="Buy", qty=1.2, signal_ts=1720000)
    a = generate_order_link_id(intent)
    b = generate_order_link_id(intent)
    assert a == b
    assert len(a) == 32



def test_order_link_id_changes_with_fields() -> None:
    a = generate_order_link_id(OrderIntent(symbol="ETCUSDT", side="Buy", qty=1.0, signal_ts=1))
    b = generate_order_link_id(OrderIntent(symbol="ETCUSDT", side="Sell", qty=1.0, signal_ts=1))
    assert a != b



def test_order_manager_place_entry() -> None:
    manager = OrderManager()
    rest = DummyRest()
    response = manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 1.0, 95.0, 110.0, 12345)
    assert response["result"]["orderStatus"] == "New"
    assert len(manager.active_orders) == 1
    record = next(iter(manager.active_orders.values()))
    assert record.symbol == "ETCUSDT"
    assert record.sl == 95.0



def test_partial_fill_handling() -> None:
    manager = OrderManager()
    rest = DummyRest()
    manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 2.0, 95.0, 110.0, 12345)
    link_id = next(iter(manager.active_orders))
    assert manager.handle_partial_fill(link_id, 1.0) is False
    assert manager.active_orders[link_id].status == "PartiallyFilled"
    assert manager.handle_partial_fill(link_id, 2.0) is True
    assert manager.active_orders[link_id].status == "Filled"



def test_order_manager_cancel_all() -> None:
    manager = OrderManager()
    rest = DummyRest()
    manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 1.0, 95.0, 110.0, 12345)
    manager.cancel_all(rest, "linear", "ETCUSDT")
    assert rest.cancel_called is True
    record = next(iter(manager.active_orders.values()))
    assert record.status == "Cancelled"



def test_update_from_ws_message() -> None:
    manager = OrderManager()
    rest = DummyRest()
    manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 1.0, 95.0, 110.0, 12345)
    link_id = next(iter(manager.active_orders))
    manager.update_from_ws_message(
        {
            "data": [
                {
                    "orderLinkId": link_id,
                    "orderId": "abc123",
                    "symbol": "ETCUSDT",
                    "side": "Buy",
                    "orderStatus": "PartiallyFilled",
                    "cumExecQty": 0.5,
                    "qty": 1.0,
                    "stopLoss": 96.0,
                    "takeProfit": 109.0,
                }
            ]
        }
    )
    record = manager.get_order(link_id)
    assert record is not None
    assert record.filled_qty == 0.5
    assert record.sl == 96.0
    assert record.tp == 109.0



def test_update_exit_order() -> None:
    manager = OrderManager()
    rest = DummyRest()
    manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 1.0, 95.0, 110.0, 12345)
    manager.update_exit_order(rest, "linear", "ETCUSDT", 0, 97.0, 111.0)
    assert rest.stop_updates[-1] == (0, 97.0, 111.0)



def test_get_active_entries_filters_filled_orders() -> None:
    manager = OrderManager()
    rest = DummyRest()
    manager.place_entry(rest, "linear", "ETCUSDT", "Buy", 1.0, 95.0, 110.0, 12345)
    link_id = next(iter(manager.active_orders))
    manager.handle_partial_fill(link_id, 1.0)
    assert manager.get_active_entries() == []
