import QRCode from "qrcode";

export async function generateCard(formData: any): Promise<string> {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  
  canvas.width = 638;
  canvas.height = 1013;
  
  // Background
  const gradient = ctx.createLinearGradient(0, 0, 638, 1013);
  gradient.addColorStop(0, "#F9FFF3");
  gradient.addColorStop(1, "#CEEDB2");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 638, 1013);
  
  // Card white background
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(40, 40, 558, 933);
  
  // Header
  ctx.fillStyle = "#084734";
  ctx.fillRect(40, 40, 558, 180);
  
  // Logo and title
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "bold 48px Inter";
  ctx.textAlign = "center";
  ctx.fillText("GREENCARD", 260, 110);
  
  ctx.font = "italic 42px serif";
  ctx.fillStyle = "#CEF17B";
  ctx.fillText("club", 400, 110);
  
  // Subtitle
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "20px Inter";
  ctx.textAlign = "center";
  ctx.fillText("CARTEIRINHA DIGITAL", 319, 160);
  ctx.fillText("CANNABIS MEDICINAL", 319, 190);
  
  // User photo
  if (formData.photo) {
    const img = new Image();
    img.src = formData.photo;
    await new Promise((resolve) => {
      img.onload = () => {
        ctx.drawImage(img, 80, 250, 180, 220);
        resolve(true);
      };
    });
  }
  
  // User information
  ctx.fillStyle = "#084734";
  ctx.font = "bold 24px Inter";
  ctx.textAlign = "left";
  ctx.fillText("NOME:", 300, 290);
  
  ctx.font = "22px Inter";
  ctx.fillText((formData.name || "").toUpperCase(), 300, 320);
  
  ctx.font = "bold 24px Inter";
  ctx.fillText("CPF:", 300, 370);
  
  ctx.font = "22px Inter";
  ctx.fillText(formData.cpf || "", 300, 400);
  
  ctx.font = "bold 24px Inter";
  ctx.fillText("NASCIMENTO:", 300, 450);
  
  ctx.font = "22px Inter";
  const date = new Date(formData.birthDate + "T00:00:00");
  const formattedDate = date.toLocaleDateString("pt-BR");
  ctx.fillText(formattedDate, 300, 480);
  
  // Authorization section
  ctx.fillStyle = "#F9FFF3";
  ctx.fillRect(60, 520, 518, 140);
  
  ctx.fillStyle = "#084734";
  ctx.font = "bold 22px Inter";
  ctx.textAlign = "center";
  ctx.fillText("AUTORIZAÇÃO VÁLIDA", 319, 560);
  
  ctx.font = "18px Inter";
  ctx.fillText("Documento digital válido em todo território nacional", 319, 590);
  ctx.fillText("conforme legislação vigente sobre cannabis medicinal", 319, 615);
  ctx.fillText("Data de emissão: " + new Date().toLocaleDateString("pt-BR"), 319, 640);
  
  // QR Code
  const qrData = JSON.stringify({
    name: formData.name,
    cpf: formData.cpf,
    date: new Date().toISOString(),
    id: Math.random().toString(36).substr(2, 9),
  });
  
  const qrDataUrl = await QRCode.toDataURL(qrData, {
    width: 200,
    margin: 1,
  });
  
  const qrImg = new Image();
  qrImg.src = qrDataUrl;
  await new Promise((resolve) => {
    qrImg.onload = () => {
      ctx.drawImage(qrImg, 219, 710, 200, 200);
      resolve(true);
    };
  });
  
  // Footer
  ctx.fillStyle = "#084734";
  ctx.font = "16px Inter";
  ctx.textAlign = "center";
  ctx.fillText("www.greencardclub.com.br", 319, 960);
  
  return canvas.toDataURL("image/png");
}