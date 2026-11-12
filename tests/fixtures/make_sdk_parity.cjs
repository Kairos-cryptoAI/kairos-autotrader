// Signs fixed test payloads with the synthetic test key using the pinned official SDK.
const { createRequire } = require("node:module");
const req = createRequire(process.argv[2] + "/package.json");
const sdk = req("@evedex/exchange-bot-sdk");
const crypto = req("@evedex/exchange-crypto");
const { ethers } = req("ethers");
(async () => {
  // The project's synthetic test key (0x11 repeated): it guards nothing and is public by construction.
  const w = new ethers.Wallet("0x" + "11".repeat(32));
  const out = { address: w.address, cases: [] };
  for (const chain of ["42161", "161803"]) {
    const wallet = new sdk.utils.Wallet({ privateKey: w.privateKey, chain });
    const market = await crypto.signMarketOrder(wallet, { id: "00474:0123456789abcdef0123456789", instrument: "SOLUSD", side: "BUY", timeInForce: "IOC", leverage: 1, cashQuantity: "16.46" });
    const close = await crypto.signPositionCloseOrder(wallet, { id: "00474:abcdef0123456789abcdef0123", instrument: "SOLUSD", leverage: 1, quantity: "0.2" });
    const tpsl = await crypto.signTpSl(wallet, { instrument: "SOLUSD", type: "stop-loss", side: "BUY", quantity: "0.2", price: "81.35", order: null });
    out.cases.push({ chain, chainType: typeof (await wallet.getChainId()), market, close, tpsl });
  }
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(String(e)); process.exit(1); });
