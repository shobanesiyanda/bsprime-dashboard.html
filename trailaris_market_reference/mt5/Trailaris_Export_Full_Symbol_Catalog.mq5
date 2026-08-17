#property script_show_inputs
#property strict

// Trailaris full broker/server/account symbol catalogue exporter.
// Read-only: exports metadata and symbol specifications; places no orders.
// Run separately on every exact prescribed MT5 broker/server/account tuple.

input string OutputPrefix = "Trailaris";

string SafeName(string value)
{
   StringReplace(value, " ", "_");
   StringReplace(value, ":", "_");
   StringReplace(value, "/", "_");
   StringReplace(value, "\\", "_");
   StringReplace(value, ".", "_");
   return value;
}

string BoolText(bool value)
{
   return value ? "true" : "false";
}

void OnStart()
{
   string server = AccountInfoString(ACCOUNT_SERVER);
   string company = AccountInfoString(ACCOUNT_COMPANY);
   string currency = AccountInfoString(ACCOUNT_CURRENCY);
   long leverage = AccountInfoInteger(ACCOUNT_LEVERAGE);
   long margin_mode = AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   long trade_mode = AccountInfoInteger(ACCOUNT_TRADE_MODE);
   bool hedge_allowed = (bool)AccountInfoInteger(ACCOUNT_HEDGE_ALLOWED);
   bool fifo_close = (bool)AccountInfoInteger(ACCOUNT_FIFO_CLOSE);

   string tag = SafeName(server);
   string catalog_file = OutputPrefix + "_BrokerSymbolCatalog_" + tag + ".csv";
   string account_file = OutputPrefix + "_BrokerAccountMetadata_" + tag + ".csv";

   int meta = FileOpen(account_file, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(meta == INVALID_HANDLE)
   {
      Print("Trailaris exporter: unable to open account metadata file. Error=", GetLastError());
      return;
   }

   FileWrite(meta,
      "exported_at_gmt","broker_company","server","account_currency","leverage",
      "margin_mode","trade_mode","hedge_allowed","fifo_close","terminal_company",
      "terminal_name","terminal_build");
   FileWrite(meta,
      TimeToString(TimeGMT(), TIME_DATE|TIME_SECONDS), company, server, currency, leverage,
      margin_mode, trade_mode, BoolText(hedge_allowed), BoolText(fifo_close),
      TerminalInfoString(TERMINAL_COMPANY), TerminalInfoString(TERMINAL_NAME),
      TerminalInfoInteger(TERMINAL_BUILD));
   FileClose(meta);

   int out = FileOpen(catalog_file, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(out == INVALID_HANDLE)
   {
      Print("Trailaris exporter: unable to open symbol catalogue file. Error=", GetLastError());
      return;
   }

   FileWrite(out,
      "symbol","description","path","currency_base","currency_profit","currency_margin",
      "trade_mode","digits","point","tick_size","tick_value","tick_value_profit",
      "tick_value_loss","contract_size","volume_min","volume_step","volume_max",
      "volume_limit","stops_level","freeze_level","margin_initial","margin_maintenance",
      "swap_mode","swap_long","swap_short","filling_mode","order_mode","expiration_mode",
      "custom","selected","visible");

   int total = SymbolsTotal(false);
   int written = 0;

   for(int i = 0; i < total; i++)
   {
      string symbol = SymbolName(i, false);
      if(symbol == "")
         continue;

      ResetLastError();

      FileWrite(out,
         symbol,
         SymbolInfoString(symbol, SYMBOL_DESCRIPTION),
         SymbolInfoString(symbol, SYMBOL_PATH),
         SymbolInfoString(symbol, SYMBOL_CURRENCY_BASE),
         SymbolInfoString(symbol, SYMBOL_CURRENCY_PROFIT),
         SymbolInfoString(symbol, SYMBOL_CURRENCY_MARGIN),
         (long)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE),
         (long)SymbolInfoInteger(symbol, SYMBOL_DIGITS),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_POINT), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE_LOSS), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_LIMIT), 12),
         (long)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL),
         (long)SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_MARGIN_INITIAL), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_MARGIN_MAINTENANCE), 12),
         (long)SymbolInfoInteger(symbol, SYMBOL_SWAP_MODE),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_SWAP_LONG), 12),
         DoubleToString(SymbolInfoDouble(symbol, SYMBOL_SWAP_SHORT), 12),
         (long)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE),
         (long)SymbolInfoInteger(symbol, SYMBOL_ORDER_MODE),
         (long)SymbolInfoInteger(symbol, SYMBOL_EXPIRATION_MODE),
         BoolText((bool)SymbolInfoInteger(symbol, SYMBOL_CUSTOM)),
         BoolText((bool)SymbolInfoInteger(symbol, SYMBOL_SELECT)),
         BoolText((bool)SymbolInfoInteger(symbol, SYMBOL_VISIBLE))
      );
      written++;
   }

   FileClose(out);

   Print("Trailaris exporter complete. server=", server,
         " symbols_total=", total,
         " symbols_written=", written,
         " catalog=", catalog_file,
         " metadata=", account_file);
}
