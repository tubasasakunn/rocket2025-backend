/**
 * SpreadsheetService - ユーザーアカウント、スコア、タイムスタンプを管理するためのサービス
 */
class SpreadsheetService {
  constructor(spreadsheetId) {
    if (!spreadsheetId) {
      // スプレッドシートIDが指定されていない場合、新しいスプレッドシートを作成
      this.spreadsheet = SpreadsheetApp.create('Rocket2025 User Data');
      
      // 新しく作成されたスプレッドシートのIDをスクリプトプロパティに保存
      const newId = this.spreadsheet.getId();
      PropertiesService.getScriptProperties().setProperty('SPREADSHEET_ID', newId);
      
      // 作成されたスプレッドシートのURLを取得
      const spreadsheetUrl = this.spreadsheet.getUrl();
      
      // IDとURLをログに出力して管理者に知らせる
      console.log('新しいスプレッドシートが作成されました。');
      console.log('ID: ' + newId);
      console.log('URL: ' + spreadsheetUrl);
      console.log('このIDをスクリプトプロパティに自動設定しました。');
      
      // ユーザーが編集権限を持っていることを確認するために、スプレッドシートを共有
      try {
        // スプレッドシートのオーナーは既にアクセス権を持っているのでここでは何もしない
        // 必要に応じて特定のユーザーとの共有を設定することも可能
      } catch (e) {
        console.log('スプレッドシートの共有設定に失敗しました: ' + e.message);
      }
    } else {
      try {
        // 指定されたIDでスプレッドシートを開く
        this.spreadsheet = SpreadsheetApp.openById(spreadsheetId);
      } catch (e) {
        // スプレッドシートが存在しないか、アクセスできない場合は新しく作成
        console.log('指定されたスプレッドシートにアクセスできません。新しいスプレッドシートを作成します。');
        this.spreadsheet = SpreadsheetApp.create('Rocket2025 User Data');
        
        // 新しく作成されたスプレッドシートのIDをスクリプトプロパティに保存
        const newId = this.spreadsheet.getId();
        PropertiesService.getScriptProperties().setProperty('SPREADSHEET_ID', newId);
        
        // 作成されたスプレッドシートのURLを取得
        const spreadsheetUrl = this.spreadsheet.getUrl();
        
        // IDとURLをログに出力して管理者に知らせる
        console.log('新しいスプレッドシートが作成されました。');
        console.log('ID: ' + newId);
        console.log('URL: ' + spreadsheetUrl);
        console.log('このIDをスクリプトプロパティに自動設定しました。');
      }
    }
    
    this.userSheet = this.spreadsheet.getSheetByName('Users') || this.createUserSheet();
  }

  /**
   * ユーザーシートが存在しない場合は作成する
   */
  createUserSheet() {
    const sheet = this.spreadsheet.insertSheet('Users');
    sheet.appendRow(['ID', 'account_name', 'score', 'update_at']);
    sheet.getRange(1, 1, 1, 4).setFontWeight('bold');
    return sheet;
  }

  /**
   * 新規ユーザーレコードを登録する
   * @param {string} account - ユーザーアカウント名
   * @param {number} score - 初期スコア
   * @return {Object} 作成されたレコードの情報
   */
  createRecord(account, score = 0) {
    // 既存のユーザーをチェック
    const existingUser = this.findByAccount(account);
    if (existingUser) {
      throw new Error(`アカウント "${account}" は既に存在します。`);
    }

    // 新規IDを生成
    const lastRow = this.userSheet.getLastRow();
    const newId = lastRow === 1 ? 1 : parseInt(this.userSheet.getRange(lastRow, 1).getValue()) + 1;
    
    // タイムスタンプを作成
    const now = new Date();
    
    // レコードを追加
    this.userSheet.appendRow([newId, account, score, now]);
    
    return {
      id: newId,
      account: account,
      score: score,
      update_at: now
    };
  }

  /**
   * すべてのユーザーレコードを取得する
   * @return {Array} ユーザーレコードの配列
   */
  getAllRecords() {
    const data = this.userSheet.getDataRange().getValues();
    const headers = data[0];
    
    // ヘッダー行をスキップ
    return data.slice(1).map((row) => {
      return {
        id: row[0],
        account: row[1],
        score: row[2],
        update_at: row[3]
      };
    });
  }

  /**
   * アカウント名でユーザーを検索する
   * @param {string} account - 検索するアカウント名
   * @return {Object|null} ユーザーレコード、見つからない場合はnull
   */
  findByAccount(account) {
    const data = this.userSheet.getDataRange().getValues();
    
    // ヘッダー行をスキップ
    for (let i = 1; i < data.length; i++) {
      if (data[i][1] === account) {
        return {
          id: data[i][0],
          account: data[i][1],
          score: data[i][2],
          update_at: data[i][3],
          rowIndex: i + 1 // スプレッドシートの行インデックスは1から始まる
        };
      }
    }
    
    return null;
  }

  /**
   * IDでユーザーを検索する
   * @param {number} id - 検索するID
   * @return {Object|null} ユーザーレコード、見つからない場合はnull
   */
  findById(id) {
    const data = this.userSheet.getDataRange().getValues();
    
    // ヘッダー行をスキップ
    for (let i = 1; i < data.length; i++) {
      if (data[i][0] == id) { // 文字列と数値の比較を許容する
        return {
          id: data[i][0],
          account: data[i][1],
          score: data[i][2],
          update_at: data[i][3],
          rowIndex: i + 1
        };
      }
    }
    
    return null;
  }

  /**
   * ユーザーのスコアを更新する
   * @param {string|number} identifier - 更新するユーザーのIDまたはアカウント
   * @param {number} newScore - 新しいスコア
   * @param {boolean} createIfNotExist - ユーザーが存在しない場合に作成するかどうか
   * @return {Object} 更新されたレコード情報
   */
  updateScore(identifier, newScore, createIfNotExist = false) {
    let user;
    
    // IDまたはアカウント名でユーザーを検索
    if (typeof identifier === 'number' || (!isNaN(identifier) && String(identifier).trim() !== '')) {
      user = this.findById(Number(identifier));
    } else if (typeof identifier === 'string' && identifier.trim() !== '') {
      user = this.findByAccount(identifier);
    }
    
    // ユーザーが見つからない場合
    if (!user) {
      if (createIfNotExist && typeof identifier === 'string' && identifier.trim() !== '') {
        // 新しいユーザーを作成
        console.log(`ユーザー "${identifier}" が見つからないため、新規作成します。`);
        return this.createRecord(identifier, newScore);
      } else {
        throw new Error(`ユーザー "${identifier}" が見つかりません。`);
      }
    }
    
    // スコアと更新日時を更新
    const now = new Date();
    this.userSheet.getRange(user.rowIndex, 3).setValue(newScore);
    this.userSheet.getRange(user.rowIndex, 4).setValue(now);
    
    return {
      id: user.id,
      account: user.account,
      score: newScore,
      update_at: now
    };
  }
}
