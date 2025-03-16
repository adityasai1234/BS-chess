let position = [
  ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
  ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
  ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
  ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

const pieceMap = {
  'r': 'rook',
  'n': 'knight',
  'b': 'bishop',
  'q': 'queen',
  'k': 'king',
  'p': 'pawn'
}

let turn = 'w';
let selectedPiece = null;

function populateBoard() {
  var board = document.getElementById('board');
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      var square = document.createElement('div');
      square.className = 'cell';
      square.classList.add((i + j) % 2 === 0 ? 'light' : 'dark');
      square.addEventListener('click', squareClicked);
      board.appendChild(square);
    }
  }
}

function placePieces() {
  var board = document.getElementById('board');
  var squares = board.getElementsByClassName('cell');
  for (let i = 0; i < 64; i++) {
    var piece = document.createElement('img');
    if (position[Math.floor(i / 8)][i % 8] !== ' ') {
      const pieceText = position[Math.floor(i / 8)][i % 8];
      const pieceIsDark = pieceText.toLowerCase() === pieceText;
      piece.src = `assets/${pieceIsDark ? 'black' : 'white'}-${pieceMap[pieceText.toLowerCase()]}.png`;
      piece.className = 'piece';
    }
    squares[i].appendChild(piece);
  }
}

// How pieces will be moved:
// 1. Click on a piece
// 2. Click on a square to move the piece to
// 3. If the move is valid, move the piece
// 4. If the move is invalid, do nothing
// 5. Change the turn
// 6. Repeat
function squareClicked() {
  const square = this;
  if (selectedPiece === null) {
    if (square.getElementsByClassName('piece').length === 0) {
      return;
    }
    selectedPiece = square;
    square.classList.add('selected');
  }
  else {
    if (selectedPiece === square) {
      selectedPiece = null;
      square.classList.remove('selected');
      return;
    }

    if (isValidMove(square, selectedPiece)) {
      const piece = selectedPiece.getElementsByClassName('piece')[0];
      square.innerHTML = '';
      square.appendChild(piece);
      selectedPiece.innerHTML = '';
      selectedPiece.classList.remove('selected');
      selectedPiece = null;
    }
  }
}

function isValidMove(square, selectedPiece) {
  const piece = selectedPiece.getElementsByClassName('piece')[0];

  const pieceText = piece.src.split('/').pop().split('.')[0].split('-')[1];

  const pieceIsDark = piece.src.includes('black');
  const pieceRow = Math.floor(Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) / 8);
  const pieceCol = Array.from(selectedPiece.parentNode.children).indexOf(selectedPiece) % 8;
  const squareRow = Math.floor(Array.from(square.parentNode.children).indexOf(square) / 8);
  const squareCol = Array.from(square.parentNode.children).indexOf(square) % 8;

  const squarePiece = square.getElementsByClassName('piece').length > 0 ? square.getElementsByClassName('piece')[0] : null;

  if (pieceText === 'pawn') {
    let offset = pieceIsDark ? 1 : -1;

    let attackingSquareLeft = pieceCol !== 0 ? document.getElementsByClassName('cell')[8 * (pieceRow + offset) + pieceCol - 1] : null;
    let attackingSquareRight = pieceCol !== 7 ? document.getElementsByClassName('cell')[8 * (pieceRow + offset) + pieceCol + 1] : null;

    if (squarePiece === null) {
      if (squareRow === pieceRow + offset && squareCol === pieceCol) {
        return true;
      }
    } else {
      return (square === attackingSquareLeft || square === attackingSquareRight) && (squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
    }
  } else if (pieceText === 'rook') {
    return (squareRow === pieceRow || squareCol === pieceCol) && 
      (squareRow !== pieceRow || squareCol !== pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'knight') {
    return (Math.abs(squareRow - pieceRow) === 2 && Math.abs(squareCol - pieceCol) === 1) || 
      (Math.abs(squareRow - pieceRow) === 1 && Math.abs(squareCol - pieceCol) === 2) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'bishop') {
    return Math.abs(squareRow - pieceRow) === Math.abs(squareCol - pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'queen') {
    return (squareRow === pieceRow || squareCol === pieceCol || Math.abs(squareRow - pieceRow) === Math.abs(squareCol - pieceCol)) && 
      (squareRow !== pieceRow || squareCol !== pieceCol) && 
      !isPieceBetween(squareRow, squareCol, pieceRow, pieceCol) && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  } else if (pieceText === 'king') {
    return Math.abs(squareRow - pieceRow) <= 1 && Math.abs(squareCol - pieceCol) <= 1 && 
      (squarePiece === null || squarePiece.src.includes(pieceIsDark ? 'white' : 'black'));
  }

  return false;
}

function isPieceBetween(row1, col1, row2, col2) {
  if (row1 === row2) {
    for (let i = Math.min(col1, col2) + 1; i < Math.max(col1, col2); i++) {
      if (document.getElementsByClassName('cell')[8 * row1 + i].getElementsByClassName('piece').length > 0) {
        return true;
      }
    }
  } else if (col1 === col2) {
    for (let i = Math.min(row1, row2) + 1; i < Math.max(row1, row2); i++) {
      if (document.getElementsByClassName('cell')[8 * i + col1].getElementsByClassName('piece').length > 0) {
        return true;
      }
    }
  }
  return false;
}

populateBoard();
placePieces();
