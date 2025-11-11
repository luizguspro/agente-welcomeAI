export function formatCPF(value: string): string {
  const cpf = value.replace(/\D/g, '');
  
  if (cpf.length <= 3) return cpf;
  if (cpf.length <= 6) return cpf.slice(0, 3) + '.' + cpf.slice(3);
  if (cpf.length <= 9) {
    return cpf.slice(0, 3) + '.' + cpf.slice(3, 6) + '.' + cpf.slice(6);
  }
  return (
    cpf.slice(0, 3) +
    '.' +
    cpf.slice(3, 6) +
    '.' +
    cpf.slice(6, 9) +
    '-' +
    cpf.slice(9, 11)
  );
}

export function formatPhone(value: string): string {
  const phone = value.replace(/\D/g, '');
  
  if (phone.length <= 2) return '(' + phone;
  if (phone.length <= 6) {
    return '(' + phone.slice(0, 2) + ') ' + phone.slice(2);
  }
  if (phone.length <= 10) {
    return (
      '(' +
      phone.slice(0, 2) +
      ') ' +
      phone.slice(2, 6) +
      '-' +
      phone.slice(6)
    );
  }
  return (
    '(' +
    phone.slice(0, 2) +
    ') ' +
    phone.slice(2, 7) +
    '-' +
    phone.slice(7, 11)
  );
}

export function formatDate(dateString: string): string {
  if (!dateString) return '';
  
  const date = new Date(dateString + 'T00:00:00');
  const day = String(date.getDate()).padStart(2, '0');
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const year = date.getFullYear();
  
  return `${day}/${month}/${year}`;
}